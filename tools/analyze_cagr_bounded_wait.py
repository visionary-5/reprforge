#!/usr/bin/env python3
"""Select bounded-wait CaGR on HR access sets, then unseal Finance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.cagr_faithful_replay import replay_cagr_comparison
from reprforge.windowed_arrival_replay import make_arrival_times
from tools.analyze_cagr_strong_adaptation import load_access_graph
from tools.analyze_windowed_arrivals import load_domain


SEEDS = (20260804, 20260805, 20260806, 20260807, 20260808)
MODELS = ("burst", "poisson")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _arrival(data: dict[str, Any], seed: int, model: str) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(data["query_count"])
    times = make_arrival_times(
        data["query_count"],
        model=model,
        seed=seed + (0 if model == "burst" else 10000),
        burst_size=32,
        burst_interval=64.0,
        poisson_mean=4.0,
    )
    return order, times


def _replay(
    data: dict[str, Any],
    *,
    seed: int,
    model: str,
    policy: str,
    config: dict[str, Any] | None = None,
    with_quality: bool,
) -> Any:
    order, times = _arrival(data, seed, model)
    gains = (
        data["quality_gain"]
        if with_quality
        else np.zeros(data["query_count"], dtype=np.float64)
    )
    config = config or {}
    return replay_cagr_comparison(
        data["cohorts"],
        order,
        times,
        gains,
        base_mean_quality=data.get("base_mean_quality", 0.0),
        corpus_pages=data["corpus_pages"],
        request_batch_size=8,
        window=64,
        policy=policy,
        cache_capacity=80,
        cagr_group_pool=int(config.get("group_pool", 64)),
        cagr_theta=float(config.get("theta", 0.5)),
        cagr_membership_rule="max",
        cagr_grouping=str(config.get("family", "threshold")),
        cagr_target_group_size=int(config.get("target_group_size", 8)),
        arrival_clock="unit",
        cagr_wait_budget=float(config.get("wait_budget", 0.0)),
        cagr_min_pending=int(config.get("min_pending", 1)),
        cagr_cross_group_fill=bool(config.get("cross_group_fill", False)),
    )


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _optional_mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _aggregate(replays: list[Any], *, with_quality: bool) -> dict[str, Any]:
    sojourn = np.concatenate(
        [np.asarray(row.sojourn_unit_time, dtype=np.float64) for row in replays]
    )
    wait = np.concatenate(
        [np.asarray(row.wait_unit_time, dtype=np.float64) for row in replays]
    )
    completion_pages = np.concatenate(
        [np.asarray(row.completion_pages, dtype=np.float64) for row in replays]
    )
    publication = np.concatenate(
        [np.asarray(row.completion_unit_cost, dtype=np.float64) for row in replays]
    )
    bypass = np.concatenate(
        [np.asarray(row.bypass_count, dtype=np.int64) for row in replays]
    )
    query_count = sum(len(row.dispatch_order) for row in replays)
    total_work = sum(row.total_unit_work for row in replays)
    group_count = sum(row.groups["count"] for row in replays)
    demand_events = sum(row.cache["demand_events"] for row in replays)
    prefetch_events = sum(row.prefetch["events"] for row in replays)
    batch_count = sum(row.request_batches["count"] for row in replays)
    cross_group_count = sum(
        row.request_batches["cross_group_count"] for row in replays
    )
    result = {
        "runs": len(replays),
        "query_publications": query_count,
        "sojourn_unit_time": _distribution(sojourn),
        "wait_unit_time": _distribution(wait),
        "completion_pages": _distribution(completion_pages),
        "publication_clock": _distribution(publication),
        "total_unit_work": total_work,
        "unit_work_per_query": total_work / query_count,
        "cache": {
            "demand_events": demand_events,
            "hits": sum(row.cache["hits"] for row in replays),
            "hit_fraction": (
                sum(row.cache["hits"] for row in replays) / demand_events
                if demand_events
                else 0.0
            ),
            "builds": sum(row.cache["builds"] for row in replays),
            "reloads": sum(row.cache["reloads"] for row in replays),
        },
        "prefetch": {
            "events": prefetch_events,
            "builds": sum(row.prefetch["builds"] for row in replays),
            "reloads": sum(row.prefetch["reloads"] for row in replays),
            "useful": sum(row.prefetch["useful"] for row in replays),
            "wasted": sum(row.prefetch["wasted"] for row in replays),
            "precision": (
                sum(row.prefetch["useful"] for row in replays) / prefetch_events
                if prefetch_events
                else None
            ),
            "unused_unit_work": sum(
                row.prefetch["unused_unit_work"] for row in replays
            ),
        },
        "groups": {
            "count": group_count,
            "singleton_count": sum(
                row.groups["singleton_count"] for row in replays
            ),
            "singleton_fraction": (
                sum(row.groups["singleton_count"] for row in replays) / group_count
                if group_count
                else None
            ),
            "mean_size": (
                sum(row.groups["size_mean"] * row.groups["count"] for row in replays)
                / group_count
                if group_count
                else None
            ),
            "max_size": (
                max(row.groups["size_max"] for row in replays) if group_count else None
            ),
        },
        "request_batches": {
            "count": batch_count,
            "mean_query_slot_utilization": float(
                np.mean(
                    [row.request_batches["query_slots_used_fraction"] for row in replays]
                )
            ),
            "mean_group_purity": _optional_mean(
                [row.request_batches["group_purity_mean"] for row in replays]
            ),
            "cross_group_count": cross_group_count,
            "cross_group_fraction": (
                cross_group_count / batch_count if batch_count else None
            ),
        },
        "starvation": {
            "count": int(np.sum(bypass >= 64)),
            "fraction": float(np.mean(bypass >= 64)),
            "max_younger_bypass": int(bypass.max()),
        },
        "final_union_pages": sorted({row.final_unique_pages for row in replays}),
        "dispatch_complete": all(
            len(row.dispatch_order) == len(set(row.dispatch_order)) for row in replays
        ),
        "order_sha256": _json_digest([row.dispatch_order for row in replays]),
    }
    if with_quality:
        result["quality_work_auc"] = _distribution(
            np.asarray([row.quality_work_auc for row in replays])
        )
        regrets = [row.normalized_quality_regret_auc for row in replays]
        result["normalized_quality_regret_auc"] = (
            _distribution(np.asarray(regrets, dtype=np.float64))
            if all(value is not None for value in regrets)
            else None
        )
    return result


def _candidate_grid() -> list[dict[str, Any]]:
    return [
        {
            "family": "fixed_jaccard",
            "target_group_size": 16,
            "group_pool": 64,
            "capacity": 80,
            "cross_group_fill": True,
            "wait_budget": wait_budget,
            "min_pending": min_pending,
        }
        for wait_budget in (0, 4, 16, 64)
        for min_pending in (4, 8, 16)
    ]


def _config_id(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _select_on_hr(access: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fifo = {
        model: _aggregate(
            [
                _replay(
                    access,
                    seed=seed,
                    model=model,
                    policy="fifo",
                    with_quality=False,
                )
                for seed in SEEDS
            ],
            with_quality=False,
        )
        for model in MODELS
    }
    rows = []
    for config in _candidate_grid():
        by_model = {}
        sojourn_ratios = []
        work_ratios = []
        p95_ratios = []
        eligible = True
        for model in MODELS:
            aggregate = _aggregate(
                [
                    _replay(
                        access,
                        seed=seed,
                        model=model,
                        policy="cagr_faithful",
                        config=config,
                        with_quality=False,
                    )
                    for seed in SEEDS
                ],
                with_quality=False,
            )
            reference = fifo[model]
            sojourn_ratio = (
                aggregate["sojourn_unit_time"]["mean"]
                / reference["sojourn_unit_time"]["mean"]
            )
            work_ratio = (
                aggregate["unit_work_per_query"]
                / reference["unit_work_per_query"]
            )
            p95_ratio = (
                aggregate["sojourn_unit_time"]["p95"]
                / reference["sojourn_unit_time"]["p95"]
            )
            union_ok = aggregate["final_union_pages"] == [
                access["candidate_union_pages"]
            ]
            eligible = bool(
                eligible
                and config["wait_budget"] in {0, 4, 16, 64}
                and union_ok
                and aggregate["dispatch_complete"]
                and p95_ratio <= 1.05
                and aggregate["starvation"]["fraction"] <= 0.05
            )
            sojourn_ratios.append(sojourn_ratio)
            work_ratios.append(work_ratio)
            p95_ratios.append(p95_ratio)
            by_model[model] = {
                "aggregate": aggregate,
                "fifo": reference,
                "mean_sojourn_ratio_vs_fifo": sojourn_ratio,
                "unit_work_ratio_vs_fifo": work_ratio,
                "p95_sojourn_ratio_vs_fifo": p95_ratio,
            }
        score = float(
            np.mean(
                [
                    0.5 * sojourn_ratio + 0.5 * work_ratio
                    for sojourn_ratio, work_ratio in zip(
                        sojourn_ratios, work_ratios
                    )
                ]
            )
        )
        rows.append(
            {
                "config": config,
                "eligible": eligible,
                "selection_score": score,
                "worst_mean_ratio": max(sojourn_ratios + work_ratios),
                "mean_p95_sojourn_ratio": float(np.mean(p95_ratios)),
                "by_arrival": by_model,
            }
        )
    deployable = [row for row in rows if row["eligible"]]
    selected_row = (
        min(
            deployable,
            key=lambda row: (
                row["selection_score"],
                row["worst_mean_ratio"],
                row["mean_p95_sojourn_ratio"],
                row["config"]["wait_budget"],
                row["config"]["min_pending"],
                _config_id(row["config"]),
            ),
        )
        if deployable
        else None
    )
    selected = None if selected_row is None else selected_row["config"]
    return rows, {
        "criterion": (
            "among finite-wait, union/query-parity candidates with each arrival "
            "P95 sojourn<=1.05*FIFO and starvation<=5%, minimize the HR "
            "burst/Poisson mean of 0.5*mean-sojourn/FIFO + "
            "0.5*unit-work-per-query/FIFO"
        ),
        "selected": selected,
        "selected_config_sha256": _json_digest(selected),
        "finance_opened_during_selection": False,
        "quality_or_visual_opened_during_selection": False,
    }


def _evaluate_full(
    data: dict[str, Any], selected: dict[str, Any] | None
) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {
        "fifo": {"policy": "fifo"},
        "current_overlap": {"policy": "overlap_only"},
        "history_popularity": {"policy": "history_popularity"},
        "static_popularity_offline": {"policy": "static_popularity"},
        "faithful_theta_0.5": {
            "policy": "cagr_faithful",
            "config": {
                "family": "threshold",
                "theta": 0.5,
                "target_group_size": 8,
                "group_pool": 64,
                "wait_budget": 0,
                "min_pending": 1,
                "cross_group_fill": False,
            },
        },
        "bounded_no_wait_anchor": {
            "policy": "cagr_faithful",
            "config": {
                "family": "fixed_jaccard",
                "target_group_size": 16,
                "group_pool": 64,
                "wait_budget": 0,
                "min_pending": 4,
                "cross_group_fill": True,
            },
        },
        "frontier": {"policy": "frontier"},
    }
    if selected is not None:
        methods["hr_selected_bounded_cagr"] = {
            "policy": "cagr_faithful",
            "config": selected,
        }
    result = {}
    for model in MODELS:
        result[model] = {}
        for name, method in methods.items():
            replays = [
                _replay(
                    data,
                    seed=seed,
                    model=model,
                    policy=method["policy"],
                    config=method.get("config"),
                    with_quality=True,
                )
                for seed in SEEDS
            ]
            result[model][name] = {
                "capacity": 80,
                "config": method.get("config"),
                "aggregate": _aggregate(replays, with_quality=True),
            }
    return result


def _finance_gate(
    finance: dict[str, Any], selected: dict[str, Any] | None
) -> dict[str, Any]:
    checks = []
    if selected is not None:
        for model in MODELS:
            adaptation = finance[model]["hr_selected_bounded_cagr"]["aggregate"]
            frontier = finance[model]["frontier"]["aggregate"]
            sojourn_adv = 1.0 - (
                frontier["sojourn_unit_time"]["mean"]
                / adaptation["sojourn_unit_time"]["mean"]
            )
            work_adv = 1.0 - (
                frontier["unit_work_per_query"]
                / adaptation["unit_work_per_query"]
            )
            adaptation_pareto = bool(
                adaptation["sojourn_unit_time"]["mean"]
                <= frontier["sojourn_unit_time"]["mean"]
                and adaptation["unit_work_per_query"]
                <= frontier["unit_work_per_query"]
                and (
                    adaptation["sojourn_unit_time"]["mean"]
                    < frontier["sojourn_unit_time"]["mean"]
                    or adaptation["unit_work_per_query"]
                    < frontier["unit_work_per_query"]
                )
            )
            checks.append(
                {
                    "arrival_model": model,
                    "frontier_sojourn_advantage": sojourn_adv,
                    "frontier_unit_work_advantage": work_adv,
                    "adaptation_pareto_dominates_frontier": adaptation_pareto,
                    "frontier_over_adaptation_p95_sojourn_ratio": (
                        frontier["sojourn_unit_time"]["p95"]
                        / adaptation["sojourn_unit_time"]["p95"]
                    ),
                    "frontier_regret_minus_adaptation": (
                        frontier["normalized_quality_regret_auc"]["mean"]
                        - adaptation["normalized_quality_regret_auc"]["mean"]
                    ),
                    "frontier_starvation_minus_adaptation": (
                        frontier["starvation"]["fraction"]
                        - adaptation["starvation"]["fraction"]
                    ),
                    "passes": bool(
                        sojourn_adv >= 0.05
                        and work_adv >= 0.05
                        and not adaptation_pareto
                    ),
                }
            )
    no_deployable = selected is None
    decision = (
        "STRONGER BASELINE SURVIVED"
        if not no_deployable and all(check["passes"] for check in checks)
        else "STOP/DOWNGRADE"
    )
    return {
        "criterion": (
            "frontier must be >=5% better than the unique HR-selected bounded "
            "CaGR in both Finance mean arrival-to-publication sojourn and "
            "charged unit work per query, for burst and Poisson"
        ),
        "checks": checks,
        "no_deployable_hr_selection": no_deployable,
        "decision": decision,
        "paper_action": (
            "stronger CaGR-style baseline survived the preregistered stress test"
            if decision == "STRONGER BASELINE SURVIVED"
            else (
                "stop: HR deployment qualification was not established"
                if no_deployable
                else "downgrade/stop stronger-baseline claim and expose Finance result"
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hr_access = load_access_graph(args.data_root / "hr")
    selection_rows, selection = _select_on_hr(hr_access)
    selected = selection["selected"]
    frozen_digest = selection["selected_config_sha256"]

    hr_full = load_domain(args.data_root / "hr", 20)
    if _json_digest(selected) != frozen_digest:
        raise AssertionError("selected configuration changed before unsealing")
    finance_full = load_domain(args.data_root / "finance", 20)
    if _json_digest(selected) != frozen_digest:
        raise AssertionError("selected configuration changed after unsealing")

    evaluation = {
        "hr_post_hoc": _evaluate_full(hr_full, selected),
        "finance_sealed": _evaluate_full(finance_full, selected),
    }
    gate = _finance_gate(evaluation["finance_sealed"], selected)
    report = {
        "schema_version": 1,
        "stage": "preregistered-bounded-wait-cagr-stress-test",
        "gpu_used": False,
        "clock_scope": (
            "arrival-to-publication unit-time: build/reload/prefetch=1, hit=0; "
            "idle and bounded wait advance elapsed time but not charged unit work"
        ),
        "hr_access_only_provenance": hr_access["provenance"],
        "selection": {
            **selection,
            "candidate_count": len(selection_rows),
            "candidate_table": selection_rows,
        },
        "unseal_audit": {
            "selected_config_sha256_before_full_load": frozen_digest,
            "selected_config_sha256_after_finance_load": _json_digest(selected),
            "hr_quality_loaded_after_selection": True,
            "finance_loaded_after_selection": True,
        },
        "inputs_after_unseal": {
            "hr": {
                "query_count": hr_full["query_count"],
                "corpus_pages": hr_full["corpus_pages"],
                "candidate_union_pages": hr_full["candidate_union_pages"],
                "provenance": hr_full["provenance"],
            },
            "finance": {
                "query_count": finance_full["query_count"],
                "corpus_pages": finance_full["corpus_pages"],
                "candidate_union_pages": finance_full["candidate_union_pages"],
                "provenance": finance_full["provenance"],
            },
        },
        "evaluation": evaluation,
        "finance_gate": gate,
        "mandatory_interpretation": (
            "this gate supersedes the no-wait infeasibility result only for the "
            "specific finite-wait fixed-Jaccard deployment family"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": selected,
                "finance_gate": gate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
