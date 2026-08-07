"""Cost and quality helpers for the defer--materialize phase diagram."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def smallest_oracle_quality_plan(
    curve: Mapping[str, Mapping[str, Any]],
    *,
    full_hit: float,
    minimum_gain_recovery: float,
    maximum_hit_loss: float,
) -> dict[str, Any] | None:
    eligible = [
        dict(row, requested_budget=float(budget))
        for budget, row in curve.items()
        if float(row["ndcg_gain_recovery"]) >= minimum_gain_recovery
        and full_hit - float(row["query_hit_at_20"]) <= maximum_hit_loss
    ]
    return min(
        eligible,
        key=lambda row: (float(row["selected_page_fraction"]), int(row["selected_pages"])),
        default=None,
    )


def history_policy_no_regression(
    curve: Mapping[str, Mapping[str, Any]], *, tolerance: float = 1e-12
) -> dict[str, Any]:
    base = float(curve["0.0"]["ndcg_at_10"]["mean"])
    rows = []
    for budget, row in sorted(curve.items(), key=lambda item: float(item[0])):
        if float(budget) == 0:
            continue
        mean = float(row["ndcg_at_10"]["mean"])
        rows.append(
            {
                "budget": float(budget),
                "future_mean_ndcg": mean,
                "delta_vs_unmaterialized_base": mean - base,
                "passes_no_regression": mean + tolerance >= base,
            }
        )
    passing = [row for row in rows if row["passes_no_regression"]]
    return {
        "base_future_mean_ndcg": base,
        "rows": rows,
        "any_nonzero_budget_passes": bool(passing),
        "best_delta_vs_base": max(
            (row["delta_vs_unmaterialized_base"] for row in rows), default=0.0
        ),
    }


def break_even_queries(build_seconds: float, page_seconds: float, avoided_pages: int) -> float:
    if build_seconds < 0 or page_seconds <= 0 or avoided_pages <= 0:
        raise ValueError("costs must be non-negative and avoided_pages must be positive")
    return build_seconds / (page_seconds * avoided_pages)


def incremental_winner(
    *,
    queries: int,
    verifier_page_seconds: float,
    avoided_pages: int,
    oracle_partial_build_seconds: float,
    current_stack_build_seconds: float,
    full_build_seconds: float,
) -> dict[str, Any]:
    costs = {
        "dvi_defer": queries * verifier_page_seconds * avoided_pages,
        "oracle_partial": oracle_partial_build_seconds,
        "current_colsmol_partial": current_stack_build_seconds,
        "full_omni": full_build_seconds,
    }
    ordered = sorted(costs, key=lambda name: (costs[name], name))
    return {"winner": ordered[0], "costs_gpu_seconds": costs}


def winner_grid(
    *,
    horizons: Sequence[int],
    avoided_pages_values: Sequence[int],
    verifier_page_seconds: float,
    oracle_partial_build_seconds: float,
    current_stack_build_seconds: float,
    full_build_seconds: float,
) -> list[dict[str, Any]]:
    return [
        {
            "queries": int(queries),
            "avoided_page_checks_per_query": int(avoided),
            **incremental_winner(
                queries=int(queries),
                verifier_page_seconds=verifier_page_seconds,
                avoided_pages=int(avoided),
                oracle_partial_build_seconds=oracle_partial_build_seconds,
                current_stack_build_seconds=current_stack_build_seconds,
                full_build_seconds=full_build_seconds,
            ),
        }
        for avoided in avoided_pages_values
        for queries in horizons
    ]
