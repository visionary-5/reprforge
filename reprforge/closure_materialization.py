"""Budgeted persistent materialization with query-scope ranking closure.

The low-cost locator is complete over the corpus.  Its candidate cohort is the
comparison unit for the high-fidelity representation: persistent pages are
reused, while every missing cohort page is constructed in a transient scratch
buffer.  Consequently storage decisions change construction work, not the
high-fidelity candidate ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ClosureCompilerConfig:
    """Frozen controls for a unit-budget closure materialization plan."""

    persistent_page_budget: int
    expected_future_queries: int
    minimum_expected_net_savings: float = 0.0

    def validate(self) -> None:
        if self.persistent_page_budget < 0:
            raise ValueError("persistent_page_budget must be nonnegative")
        if self.expected_future_queries <= 0:
            raise ValueError("expected_future_queries must be positive")


def compile_closure_plan(
    candidate_indices: np.ndarray,
    history_queries: Sequence[int],
    *,
    persistent_build_cost: np.ndarray | None = None,
    transient_build_cost: np.ndarray | None = None,
    config: ClosureCompilerConfig,
) -> dict[str, Any]:
    """Select pages whose expected avoided transient work repays persistence.

    With a page-count budget, independent page construction, and a fixed
    candidate surface, sorting by expected net savings is the exact optimizer.
    No relevance labels or future candidate rows are read.
    """

    config.validate()
    candidates = np.asarray(candidate_indices, dtype=np.int32)
    if candidates.ndim != 2 or candidates.shape[0] == 0 or candidates.shape[1] == 0:
        raise ValueError("candidate_indices must have shape [queries, depth]")
    pages = int(candidates.max()) + 1
    queries = np.asarray(list(map(int, history_queries)), dtype=np.int32)
    if queries.size == 0 or queries.min() < 0 or queries.max() >= candidates.shape[0]:
        raise ValueError("history query positions are empty or outside the surface")
    if len(set(queries.tolist())) != len(queries):
        raise ValueError("history query positions contain duplicates")
    build = _cost_vector(pages, persistent_build_cost, "persistent_build_cost")
    transient = _cost_vector(pages, transient_build_cost, "transient_build_cost")

    history_events = np.zeros(pages, dtype=np.int64)
    for query in queries:
        history_events[candidates[int(query)]] += 1
    expected_events = (
        history_events.astype(np.float64)
        * float(config.expected_future_queries)
        / float(len(queries))
    )
    expected_avoided_work = expected_events * transient
    expected_net_savings = expected_avoided_work - build
    page_ids = np.arange(pages, dtype=np.int32)
    order = page_ids[np.lexsort((page_ids, -expected_net_savings))]
    eligible = order[
        expected_net_savings[order] > config.minimum_expected_net_savings
    ]
    selected = eligible[: min(config.persistent_page_budget, len(eligible))]
    return {
        "persistent_pages": list(map(int, selected)),
        "persistent_page_count": int(len(selected)),
        "persistent_page_budget": int(config.persistent_page_budget),
        "expected_future_queries": int(config.expected_future_queries),
        "history_queries": int(len(queries)),
        "candidate_depth": int(candidates.shape[1]),
        "expected_persistent_build_work": float(build[selected].sum()),
        "expected_avoided_transient_work": float(
            expected_avoided_work[selected].sum()
        ),
        "expected_net_savings": float(expected_net_savings[selected].sum()),
        "page_statistics": [
            {
                "page": int(page),
                "history_candidate_events": int(history_events[page]),
                "expected_future_candidate_events": float(expected_events[page]),
                "expected_net_savings": float(expected_net_savings[page]),
            }
            for page in selected
        ],
        "protocol": {
            "future_candidate_rows_visible": False,
            "future_relevance_visible": False,
            "ranking_semantics": "complete candidate closure",
            "unmaterialized_candidate_action": "transient high-fidelity construction",
        },
    }


def plan_query_closure(
    candidate_pages: Sequence[int], persistent_pages: Sequence[int]
) -> dict[str, Any]:
    """Partition one locator cohort without changing its comparison support."""

    candidates = list(dict.fromkeys(map(int, candidate_pages)))
    if not candidates:
        raise ValueError("candidate_pages must not be empty")
    persistent = set(map(int, persistent_pages))
    hits = [page for page in candidates if page in persistent]
    misses = [page for page in candidates if page not in persistent]
    return {
        "candidate_pages": candidates,
        "persistent_pages": hits,
        "transient_pages": misses,
        "candidate_closure_complete": True,
        "ranking_quality_invariant_to_storage_state": True,
    }


def _cost_vector(pages: int, values: np.ndarray | None, label: str) -> np.ndarray:
    output = (
        np.ones(pages, dtype=np.float64)
        if values is None
        else np.asarray(values, dtype=np.float64)
    )
    if output.shape != (pages,) or not np.all(np.isfinite(output)):
        raise ValueError(f"{label} must contain one finite value per page")
    if np.any(output <= 0.0):
        raise ValueError(f"{label} must be positive")
    return output
