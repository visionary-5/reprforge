"""Event-driven causal scheduler for hard-budget progressive compilation.

The scheduler deliberately has no quality-gain or future-arrival interface.
An external replay or serving loop delivers one observed arrival/timer event at
a time and synchronizes the currently compiled pages and active LRU state.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Any


class HardBudgetFrontier:
    """Fixed B32 completion/deadline scheduler with a causal event API."""

    POLICY_NAME = "hard_budget_frontier"
    BYPASS_BUDGET = 32
    COMPLETION_WEIGHT = 0.75
    DEADLINE_WEIGHT = 0.25
    DEADLINE_SCALE = 256.0
    TIMEOUT = 16.0
    COST_EPSILON = 1.0
    BATCH_SIZE = 8
    WINDOW = 64
    CACHE_CAPACITY = 80

    def __init__(self) -> None:
        self.request_batch_size = self.BATCH_SIZE
        self.window = self.WINDOW
        self.cache_capacity = self.CACHE_CAPACITY
        self._pending: list[int] = []
        self._cohorts: dict[int, frozenset[int]] = {}
        self._arrival_time: dict[int, float] = {}
        self._arrival_rank: dict[int, int] = {}
        self._bypass: dict[int, int] = {}
        self._compiled: set[int] = set()
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._last_event_time = 0.0
        self._protected_queries: set[int] = set()
        self._arrival_events = 0
        self._timer_events = 0
        self._dispatch_events = 0
        self._selection_count = 0
        self._forced_selection_count = 0
        self._utility_evaluations = 0
        self._page_probes = 0
        self._feasibility_comparisons = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def _observe_time(self, event_time: float) -> float:
        now = float(event_time)
        if now + 1e-12 < self._last_event_time:
            raise ValueError("events must be delivered in non-decreasing time")
        self._last_event_time = max(self._last_event_time, now)
        return now

    def arrival(
        self, query_id: int, locator_cohort: Iterable[int], event_time: float
    ) -> None:
        """Deliver one observed query arrival; no future trace is accepted."""

        now = self._observe_time(event_time)
        query = int(query_id)
        cohort = frozenset(int(page) for page in locator_cohort)
        if not cohort:
            raise ValueError("locator cohort must be non-empty")
        if query in self._cohorts:
            raise ValueError(f"duplicate query arrival: {query}")
        self._cohorts[query] = cohort
        self._arrival_time[query] = now
        self._arrival_rank[query] = self._arrival_events
        self._bypass[query] = 0
        self._pending.append(query)
        self._arrival_events += 1

    def sync_index_state(
        self, compiled_pages: Iterable[int], active_lru_oldest_first: Sequence[int]
    ) -> None:
        """Synchronize only the currently observable representation state."""

        compiled = {int(page) for page in compiled_pages}
        lru_values = [int(page) for page in active_lru_oldest_first]
        if len(lru_values) != len(set(lru_values)):
            raise ValueError("active LRU pages must be unique")
        if len(lru_values) > self.cache_capacity:
            raise ValueError("active LRU exceeds configured capacity")
        if not set(lru_values).issubset(compiled):
            raise ValueError("active LRU must be a subset of compiled pages")
        self._compiled = compiled
        self._lru = OrderedDict((page, None) for page in lru_values)

    def timer_deadline(self) -> float | None:
        if not self._pending or len(self._pending) >= self.request_batch_size:
            return None
        oldest = min(self._pending, key=self._arrival_rank.__getitem__)
        return self._arrival_time[oldest] + self.TIMEOUT

    def ready(self, event_time: float) -> bool:
        now = self._observe_time(event_time)
        if not self._pending:
            return False
        if len(self._pending) >= self.request_batch_size:
            return True
        deadline = self.timer_deadline()
        return deadline is not None and now >= deadline - 1e-12

    def timer(self, event_time: float) -> bool:
        """Deliver an armed timer event and report whether dispatch is ready."""

        self._timer_events += 1
        return self.ready(event_time)

    def _simulate_query(
        self,
        query: int,
        compiled: set[int],
        lru: OrderedDict[int, None],
        *,
        mutate: bool,
    ) -> tuple[float, set[int], OrderedDict[int, None]]:
        candidate_compiled = compiled if mutate else set(compiled)
        candidate_lru = lru if mutate else OrderedDict(lru)
        cost = 0.0
        for page in sorted(self._cohorts[query]):
            self._page_probes += 1
            if page not in candidate_lru:
                cost += 1.0
                candidate_compiled.add(page)
            candidate_lru[page] = None
            candidate_lru.move_to_end(page)
            if len(candidate_lru) > self.cache_capacity:
                candidate_lru.popitem(last=False)
        return cost, candidate_compiled, candidate_lru

    def dispatch(self, event_time: float) -> tuple[int, ...]:
        """Select one atomic batch from currently arrived queries only."""

        now = self._observe_time(event_time)
        if not self.ready(now):
            return ()
        self._dispatch_events += 1
        virtual_compiled = set(self._compiled)
        virtual_lru = OrderedDict(self._lru)
        batch: list[int] = []
        while self._pending and len(batch) < self.request_batch_size:
            # Match the serving window: every freed batch slot admits the next
            # oldest pending query into the visible prefix.
            visible = self._pending[: min(self.window, len(self._pending))]
            completion_density: dict[int, float] = {}
            for query in visible:
                marginal_cost, _, _ = self._simulate_query(
                    query, virtual_compiled, virtual_lru, mutate=False
                )
                completion_density[query] = 1.0 / max(
                    marginal_cost, self.COST_EPSILON
                )
                self._utility_evaluations += 1
            completion_scale = max(completion_density.values(), default=1.0)
            utility: dict[int, float] = {}
            for query in visible:
                completion_term = completion_density[query] / completion_scale
                age = max(0.0, now - self._arrival_time[query])
                deadline_term = min(age / self.DEADLINE_SCALE, 1.0)
                utility[query] = (
                    self.COMPLETION_WEIGHT * completion_term
                    + self.DEADLINE_WEIGHT * deadline_term
                )
            unconstrained = min(
                visible,
                key=lambda query: (
                    -utility[query],
                    self._arrival_rank[query],
                    query,
                ),
            )

            def feasible(query: int) -> bool:
                for victim in self._pending:
                    if self._arrival_rank[victim] < self._arrival_rank[query]:
                        self._feasibility_comparisons += 1
                        if self._bypass[victim] + 1 > self.BYPASS_BUDGET:
                            return False
                return True

            feasible_queries = [query for query in visible if feasible(query)]
            if not feasible_queries:
                raise AssertionError("oldest pending query must remain feasible")
            selected = min(
                feasible_queries,
                key=lambda query: (
                    -utility[query],
                    self._arrival_rank[query],
                    query,
                ),
            )
            self._selection_count += 1
            if selected != unconstrained:
                self._forced_selection_count += 1
                self._protected_queries.update(
                    victim
                    for victim in self._pending
                    if self._arrival_rank[victim]
                    < self._arrival_rank[unconstrained]
                    and self._bypass[victim] + 1 > self.BYPASS_BUDGET
                )
            for victim in self._pending:
                if self._arrival_rank[victim] < self._arrival_rank[selected]:
                    self._bypass[victim] += 1
            self._pending.remove(selected)
            batch.append(selected)
            _, virtual_compiled, virtual_lru = self._simulate_query(
                selected, virtual_compiled, virtual_lru, mutate=True
            )
        return tuple(batch)

    def bypass_counts(self, query_ids: Sequence[int]) -> tuple[int, ...]:
        return tuple(self._bypass[int(query)] for query in query_ids)

    def audit(self) -> dict[str, Any]:
        selections = self._selection_count
        max_bypass = max(self._bypass.values(), default=0)
        return {
            "policy": self.POLICY_NAME,
            "counting_scope": (
                "detailed utility+page-probe+feasibility+event operation proxy"
            ),
            "fixed_config": {
                "bypass_budget": self.BYPASS_BUDGET,
                "completion_weight": self.COMPLETION_WEIGHT,
                "deadline_weight": self.DEADLINE_WEIGHT,
                "deadline_scale": self.DEADLINE_SCALE,
                "timeout": self.TIMEOUT,
                "batch_size": self.request_batch_size,
                "window": self.window,
                "cache_capacity": self.cache_capacity,
            },
            "arrival_events": self._arrival_events,
            "timer_events": self._timer_events,
            "dispatch_events": self._dispatch_events,
            "selection_count": selections,
            "forced_selection_count": self._forced_selection_count,
            "forced_selection_fraction": (
                self._forced_selection_count / selections if selections else 0.0
            ),
            "protected_unique_queries": len(self._protected_queries),
            "max_younger_bypass": max_bypass,
            "budget_violation_count": sum(
                value > self.BYPASS_BUDGET for value in self._bypass.values()
            ),
            "utility_evaluations": self._utility_evaluations,
            "page_probes": self._page_probes,
            "feasibility_comparisons": self._feasibility_comparisons,
            "control_operations": (
                self._utility_evaluations
                + self._page_probes
                + self._feasibility_comparisons
                + self._arrival_events
                + self._timer_events
                + self._dispatch_events
            ),
        }
