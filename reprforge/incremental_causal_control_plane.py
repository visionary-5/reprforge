"""Exact instrumented and incremental implementations of the B32 scheduler."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from reprforge.causal_hard_frontier import HardBudgetFrontier


class TransparentInstrumentedHardBudgetFrontier(HardBudgetFrontier):
    """The reference implementation with a shared logical-item counter."""

    POLICY_NAME = "hard_budget_frontier_transparent"

    def __init__(self) -> None:
        super().__init__()
        self._state_copy_items = 0
        self._cohort_order_items = 0

    def _simulate_query(
        self,
        query: int,
        compiled: set[int],
        lru: OrderedDict[int, None],
        *,
        mutate: bool,
    ) -> tuple[float, set[int], OrderedDict[int, None]]:
        if not mutate:
            self._state_copy_items += len(compiled) + len(lru)
        self._cohort_order_items += len(self._cohorts[query])
        return super()._simulate_query(query, compiled, lru, mutate=mutate)

    def _retained_state_items(self) -> int:
        return (
            sum(len(cohort) for cohort in self._cohorts.values())
            + len(self._compiled)
            + len(self._lru)
            + 4 * len(self._pending)
        )

    def audit(self) -> dict[str, Any]:
        result = super().audit()
        result["policy"] = self.POLICY_NAME
        result["operation_proxy_v2"] = {
            "definition": "logical state/page/feasibility/event items",
            "base_control_operations": result["control_operations"],
            "state_copy_items": self._state_copy_items,
            "cohort_order_items": self._cohort_order_items,
            "frontier_comparisons": 0,
            "total": (
                result["control_operations"]
                + self._state_copy_items
                + self._cohort_order_items
            ),
        }
        result["retained_state_items"] = self._retained_state_items()
        return result


class IncrementalHardBudgetFrontier(HardBudgetFrontier):
    """Exact B32 scheduler without compiled copies or quadratic feasibility."""

    POLICY_NAME = "hard_budget_frontier_incremental"

    def __init__(self) -> None:
        super().__init__()
        self._sorted_cohorts: dict[int, tuple[int, ...]] = {}
        self._state_copy_items = 0
        self._cohort_order_items = 0
        self._frontier_comparisons = 0

    def arrival(
        self, query_id: int, locator_cohort: Iterable[int], event_time: float
    ) -> None:
        cohort = tuple(int(page) for page in locator_cohort)
        super().arrival(query_id, cohort, event_time)
        ordered = tuple(sorted(set(cohort)))
        self._sorted_cohorts[int(query_id)] = ordered
        self._cohort_order_items += len(ordered)

    def _marginal_cost(
        self, query: int, lru: OrderedDict[int, None]
    ) -> float:
        candidate_lru = OrderedDict(lru)
        self._state_copy_items += len(candidate_lru)
        cost = 0.0
        for page in self._sorted_cohorts[query]:
            self._page_probes += 1
            if page not in candidate_lru:
                cost += 1.0
            candidate_lru[page] = None
            candidate_lru.move_to_end(page)
            if len(candidate_lru) > self.cache_capacity:
                candidate_lru.popitem(last=False)
        return cost

    def _apply_query(
        self,
        query: int,
        compiled: set[int],
        lru: OrderedDict[int, None],
    ) -> None:
        for page in self._sorted_cohorts[query]:
            self._page_probes += 1
            if page not in lru:
                compiled.add(page)
            lru[page] = None
            lru.move_to_end(page)
            if len(lru) > self.cache_capacity:
                lru.popitem(last=False)

    def _visible_costs(
        self,
        visible: list[int],
        virtual_lru: OrderedDict[int, None],
    ) -> dict[int, float]:
        costs = {}
        for query in visible:
            costs[query] = self._marginal_cost(query, virtual_lru)
            self._utility_evaluations += 1
        return costs

    def _feasible_visible(self, visible: list[int]) -> list[int]:
        head = self._pending[0]
        self._frontier_comparisons += 1
        return [head] if self._bypass[head] >= self.BYPASS_BUDGET else visible

    def dispatch(self, event_time: float) -> tuple[int, ...]:
        now = self._observe_time(event_time)
        if not self.ready(now):
            return ()
        self._dispatch_events += 1
        virtual_compiled = set(self._compiled)
        virtual_lru = OrderedDict(self._lru)
        batch: list[int] = []
        while self._pending and len(batch) < self.request_batch_size:
            visible = self._pending[: min(self.window, len(self._pending))]
            costs = self._visible_costs(visible, virtual_lru)
            completion_density = {
                query: 1.0 / max(costs[query], self.COST_EPSILON)
                for query in visible
            }
            completion_scale = max(completion_density.values(), default=1.0)
            utility = {}
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
            feasible = self._feasible_visible(visible)
            selected = min(
                feasible,
                key=lambda query: (
                    -utility[query],
                    self._arrival_rank[query],
                    query,
                ),
            )
            self._selection_count += 1
            if selected != unconstrained:
                self._forced_selection_count += 1
                for victim in self._pending:
                    self._frontier_comparisons += 1
                    if (
                        self._arrival_rank[victim]
                        < self._arrival_rank[unconstrained]
                        and self._bypass[victim] + 1 > self.BYPASS_BUDGET
                    ):
                        self._protected_queries.add(victim)
            for victim in self._pending:
                if self._arrival_rank[victim] < self._arrival_rank[selected]:
                    self._bypass[victim] += 1
            self._pending.remove(selected)
            batch.append(selected)
            self._apply_query(selected, virtual_compiled, virtual_lru)
        return tuple(batch)

    def _retained_state_items(self) -> int:
        return (
            sum(len(cohort) for cohort in self._cohorts.values())
            + sum(len(cohort) for cohort in self._sorted_cohorts.values())
            + len(self._compiled)
            + len(self._lru)
            + 4 * len(self._pending)
        )

    def audit(self) -> dict[str, Any]:
        result = super().audit()
        result["policy"] = self.POLICY_NAME
        result["operation_proxy_v2"] = {
            "definition": "logical state/page/feasibility/event items",
            "base_control_operations": (
                result["control_operations"]
                - result["feasibility_comparisons"]
            ),
            "state_copy_items": self._state_copy_items,
            "cohort_order_items": self._cohort_order_items,
            "frontier_comparisons": self._frontier_comparisons,
            "total": (
                result["control_operations"]
                - result["feasibility_comparisons"]
                + self._state_copy_items
                + self._cohort_order_items
                + self._frontier_comparisons
            ),
        }
        result["retained_state_items"] = self._retained_state_items()
        result["optimization"] = {
            "cached_sorted_cohorts": True,
            "candidate_compiled_copy": False,
            "uniform_budget_head_frontier": True,
        }
        return result


class EfficientDelaySchedulingD32(IncrementalHardBudgetFrontier):
    """Efficient query/page Delay-D32 control-plane comparator."""

    POLICY_NAME = "delay_scheduling_d32_control"

    def dispatch(self, event_time: float) -> tuple[int, ...]:
        now = self._observe_time(event_time)
        if not self.ready(now):
            return ()
        self._dispatch_events += 1
        virtual_compiled = set(self._compiled)
        virtual_lru = OrderedDict(self._lru)
        batch: list[int] = []
        while self._pending and len(batch) < self.request_batch_size:
            visible = self._pending[: min(self.window, len(self._pending))]
            costs = self._visible_costs(visible, virtual_lru)
            active = set(virtual_lru)
            locality_best = min(
                visible,
                key=lambda query: (
                    costs[query],
                    -len(self._cohorts[query] & active),
                    self._arrival_rank[query],
                    query,
                ),
            )
            feasible = self._feasible_visible(visible)
            selected = locality_best if locality_best in feasible else feasible[0]
            self._selection_count += 1
            if selected != locality_best:
                self._forced_selection_count += 1
                self._protected_queries.add(self._pending[0])
            for victim in self._pending:
                if self._arrival_rank[victim] < self._arrival_rank[selected]:
                    self._bypass[victim] += 1
            self._pending.remove(selected)
            batch.append(selected)
            self._apply_query(selected, virtual_compiled, virtual_lru)
        return tuple(batch)

    def audit(self) -> dict[str, Any]:
        result = super().audit()
        result["policy"] = self.POLICY_NAME
        result["adaptation_not_official_implementation"] = True
        result["fixed_config"]["locality_rule"] = (
            "min_exact_demand_cost_then_active_overlap"
        )
        return result


CONTROL_SCHEDULER_CLASSES = {
    scheduler.POLICY_NAME: scheduler
    for scheduler in (
        TransparentInstrumentedHardBudgetFrontier,
        IncrementalHardBudgetFrontier,
        EfficientDelaySchedulingD32,
    )
}
