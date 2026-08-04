#!/usr/bin/env python3
"""Select strong CaGR adaptations on HR access sets, then unseal Finance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.cagr_faithful_replay import replay_cagr_comparison
from reprforge.progressive_oracle import rank_order
from reprforge.windowed_arrival_replay import make_arrival_times
from tools.analyze_windowed_arrivals import load_domain


SEEDS = (20260804, 20260805, 20260806, 20260807, 20260808)
MODELS = ("burst", "poisson")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_access_graph(root: Path, candidate_k: int = 20) -> dict[str, Any]:
    """Load only BM25 runtime arrays; never open oracle labels or visual files."""

    text_root = root / "bm25"
    manifest_path = text_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    runtime_path = text_root / manifest["runtime_file"]
    if _sha256(runtime_path) != manifest["runtime_sha256"]:
        raise ValueError("BM25 runtime digest mismatch")
    with np.load(runtime_path, allow_pickle=False) as payload:
        scores = np.asarray(payload["scores"], dtype=np.float64)
        corpus_ids = np.asarray(payload["corpus_ids"])
        query_ids = np.asarray(payload["query_ids"])
    order = rank_order(scores, corpus_ids)
    cohorts = order[:, :candidate_k].tolist()
    return {
        "cohorts": cohorts,
        "query_count": len(query_ids),
        "corpus_pages": len(corpus_ids),
        "candidate_union_pages": len(set(order[:, :candidate_k].reshape(-1))),
        "provenance": {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "runtime_path": str(runtime_path.resolve()),
            "runtime_sha256": _sha256(runtime_path),
            "opened_files": [str(manifest_path.resolve()), str(runtime_path.resolve())],
            "oracle_or_visual_files_opened": False,
        },
    }


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
    capacity: int,
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
        cache_capacity=capacity,
        cagr_group_pool=int(config.get("group_pool", 64)),
        cagr_theta=float(config.get("theta", 0.5)),
        cagr_membership_rule="max",
        cagr_grouping=str(config.get("family", "threshold")),
        cagr_target_group_size=int(config.get("target_group_size", 8)),
    )


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _aggregate(replays: list[Any], *, with_quality: bool) -> dict[str, Any]:
    pages = np.concatenate(
        [np.asarray(row.completion_pages, dtype=np.float64) for row in replays]
    )
    costs = np.concatenate(
        [np.asarray(row.completion_unit_cost, dtype=np.float64) for row in replays]
    )
    bypass = np.concatenate(
        [np.asarray(row.bypass_count, dtype=np.int64) for row in replays]
    )
    group_count = sum(row.groups["count"] for row in replays)
    demand_events = sum(row.cache["demand_events"] for row in replays)
    prefetch_events = sum(row.prefetch["events"] for row in replays)
    result = {
        "runs": len(replays),
        "completion_pages": _distribution(pages),
        "completion_unit_cost": _distribution(costs),
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
            "count": sum(row.request_batches["count"] for row in replays),
            "mean_query_slot_utilization": float(
                np.mean(
                    [row.request_batches["query_slots_used_fraction"] for row in replays]
                )
            ),
        },
        "starvation": {
            "count": int(np.sum(bypass >= 64)),
            "fraction": float(np.mean(bypass >= 64)),
            "max_younger_bypass": int(bypass.max()),
        },
        "final_union_pages": sorted({row.final_unique_pages for row in replays}),
        "order_sha256": _json_digest([row.dispatch_order for row in replays]),
    }
    if with_quality:
        result["quality_work_auc"] = _distribution(
            np.asarray([row.quality_work_auc for row in replays])
        )
        result["normalized_quality_regret_auc"] = _distribution(
            np.asarray([row.normalized_quality_regret_auc for row in replays])
        )
    return result


def _config_id(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _candidate_grid() -> list[dict[str, Any]]:
    threshold = [
        {
            "family": "threshold",
            "theta": theta,
            "group_pool": pool,
            "capacity": capacity,
            "target_group_size": 8,
        }
        for theta in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
        for pool in (20, 40, 64)
        for capacity in (40, 80, 160)
    ]
    fixed = [
        {
            "family": "fixed_jaccard",
            "theta": 0.5,
            "group_pool": pool,
            "capacity": capacity,
            "target_group_size": size,
        }
        for size in (4, 8, 16)
        for pool in (20, 40, 64)
        for capacity in (40, 80, 160)
    ]
    return threshold + fixed


def _select_on_hr(access: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fifo: dict[tuple[int, str], dict[str, Any]] = {}
    for capacity in (40, 80, 160):
        for model in MODELS:
            fifo[(capacity, model)] = _aggregate(
                [
                    _replay(
                        access,
                        seed=seed,
                        model=model,
                        policy="fifo",
                        capacity=capacity,
                        with_quality=False,
                    )
                    for seed in SEEDS
                ],
                with_quality=False,
            )

    rows = []
    for config in _candidate_grid():
        by_model = {}
        page_ratios = []
        cost_ratios = []
        p95_cost_ratios = []
        eligible = True
        for model in MODELS:
            aggregate = _aggregate(
                [
                    _replay(
                        access,
                        seed=seed,
                        model=model,
                        policy="cagr_faithful",
                        capacity=config["capacity"],
                        config=config,
                        with_quality=False,
                    )
                    for seed in SEEDS
                ],
                with_quality=False,
            )
            reference = fifo[(config["capacity"], model)]
            page_ratio = (
                aggregate["completion_pages"]["mean"]
                / reference["completion_pages"]["mean"]
            )
            cost_ratio = (
                aggregate["completion_unit_cost"]["mean"]
                / reference["completion_unit_cost"]["mean"]
            )
            page_ratios.append(page_ratio)
            cost_ratios.append(cost_ratio)
            p95_cost_ratios.append(
                aggregate["completion_unit_cost"]["p95"]
                / reference["completion_unit_cost"]["p95"]
            )
            singleton = aggregate["groups"]["singleton_fraction"]
            utilization = aggregate["request_batches"][
                "mean_query_slot_utilization"
            ]
            union_ok = aggregate["final_union_pages"] == [
                access["candidate_union_pages"]
            ]
            eligible = bool(
                eligible
                and singleton is not None
                and singleton <= 0.50
                and utilization >= 0.50
                and union_ok
            )
            by_model[model] = {
                "aggregate": aggregate,
                "fifo_same_capacity": reference,
                "mean_page_ratio_vs_fifo": page_ratio,
                "mean_unit_cost_ratio_vs_fifo": cost_ratio,
                "p95_unit_cost_ratio_vs_fifo": p95_cost_ratios[-1],
            }
        score = float(np.mean([(p + c) / 2.0 for p, c in zip(page_ratios, cost_ratios)]))
        worst = max(page_ratios + cost_ratios)
        mean_p95 = float(np.mean(p95_cost_ratios))
        mean_hit = float(
            np.mean([by_model[model]["aggregate"]["cache"]["hit_fraction"] for model in MODELS])
        )
        rows.append(
            {
                "config": config,
                "eligible": eligible,
                "selection_score": score,
                "worst_mean_ratio": worst,
                "mean_p95_unit_cost_ratio": mean_p95,
                "mean_cache_hit_fraction": mean_hit,
                "by_arrival": by_model,
            }
        )

    def choose(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        deployable = [row for row in candidates if row["eligible"]]
        return (
            min(
                deployable,
                key=lambda row: (
                    row["selection_score"],
                    row["worst_mean_ratio"],
                    row["mean_p95_unit_cost_ratio"],
                    -row["mean_cache_hit_fraction"],
                    _config_id(row["config"]),
                ),
            )
            if deployable
            else None
        )

    selected_rows = {
        "lower_theta": choose(
            [
                row
                for row in rows
                if row["config"]["family"] == "threshold"
                and row["config"]["theta"] < 0.5
            ]
        ),
        "fixed_size": choose(
            [row for row in rows if row["config"]["family"] == "fixed_jaccard"]
        ),
        "overall": choose(rows),
    }
    selected = {
        name: (None if row is None else row["config"])
        for name, row in selected_rows.items()
    }
    return rows, {
        "criterion": (
            "minimize HR burst/Poisson mean of 0.5*completion/FIFO + "
            "0.5*unit-cost/FIFO among singleton<=50%, utilization>=50% candidates; "
            "the lower-theta deployment selection excludes the faithful theta=0.5 anchor"
        ),
        "selected": selected,
        "selected_config_sha256": _json_digest(selected),
        "finance_opened_during_selection": False,
        "quality_or_visual_opened_during_selection": False,
    }


def _evaluate_full(
    data: dict[str, Any], selected: dict[str, dict[str, Any] | None]
) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {
        "fifo": {"policy": "fifo", "capacity": 80},
        "current_overlap": {"policy": "overlap_only", "capacity": 80},
        "history_popularity": {"policy": "history_popularity", "capacity": 80},
        "static_popularity_offline": {"policy": "static_popularity", "capacity": 80},
        "faithful_theta_0.5": {
            "policy": "cagr_faithful",
            "capacity": 80,
            "config": {
                "family": "threshold",
                "theta": 0.5,
                "group_pool": 64,
                "capacity": 80,
                "target_group_size": 8,
            },
        },
        "frontier_capacity80": {"policy": "frontier", "capacity": 80},
    }
    for alias, config in selected.items():
        if config is not None:
            methods[f"hr_selected_{alias}"] = {
                "policy": "cagr_faithful",
                "capacity": config["capacity"],
                "config": config,
            }
            methods[f"frontier_for_{alias}"] = {
                "policy": "frontier",
                "capacity": config["capacity"],
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
                    capacity=method["capacity"],
                    config=method.get("config"),
                    with_quality=True,
                )
                for seed in SEEDS
            ]
            result[model][name] = {
                "capacity": method["capacity"],
                "config": method.get("config"),
                "aggregate": _aggregate(replays, with_quality=True),
            }
    return result


def _finance_gate(
    finance: dict[str, Any], selected: dict[str, dict[str, Any] | None]
) -> dict[str, Any]:
    checks = []
    unique: dict[str, tuple[str, dict[str, Any]]] = {}
    for alias, config in selected.items():
        if config is not None:
            unique.setdefault(_config_id(config), (alias, config))
    for alias, config in unique.values():
        for model in MODELS:
            adaptation = finance[model][f"hr_selected_{alias}"]["aggregate"]
            frontier = finance[model][f"frontier_for_{alias}"]["aggregate"]
            page_adv = 1.0 - (
                frontier["completion_pages"]["mean"]
                / adaptation["completion_pages"]["mean"]
            )
            cost_adv = 1.0 - (
                frontier["completion_unit_cost"]["mean"]
                / adaptation["completion_unit_cost"]["mean"]
            )
            adaptation_pareto = bool(
                adaptation["completion_pages"]["mean"]
                <= frontier["completion_pages"]["mean"]
                and adaptation["completion_unit_cost"]["mean"]
                <= frontier["completion_unit_cost"]["mean"]
                and (
                    adaptation["completion_pages"]["mean"]
                    < frontier["completion_pages"]["mean"]
                    or adaptation["completion_unit_cost"]["mean"]
                    < frontier["completion_unit_cost"]["mean"]
                )
            )
            passes = bool(page_adv >= 0.05 and cost_adv >= 0.05 and not adaptation_pareto)
            checks.append(
                {
                    "selection_alias": alias,
                    "config": config,
                    "arrival_model": model,
                    "frontier_page_advantage": page_adv,
                    "frontier_unit_cost_advantage": cost_adv,
                    "adaptation_pareto_dominates_frontier": adaptation_pareto,
                    "frontier_over_adaptation_p95_page_ratio": (
                        frontier["completion_pages"]["p95"]
                        / adaptation["completion_pages"]["p95"]
                    ),
                    "frontier_over_adaptation_p95_cost_ratio": (
                        frontier["completion_unit_cost"]["p95"]
                        / adaptation["completion_unit_cost"]["p95"]
                    ),
                    "frontier_regret_minus_adaptation": (
                        frontier["normalized_quality_regret_auc"]["mean"]
                        - adaptation["normalized_quality_regret_auc"]["mean"]
                    ),
                    "frontier_starvation_minus_adaptation": (
                        frontier["starvation"]["fraction"]
                        - adaptation["starvation"]["fraction"]
                    ),
                    "passes": passes,
                }
            )
    decision = "STRONG GO" if checks and all(row["passes"] for row in checks) else "STOP/DOWNGRADE"
    return {
        "criterion": (
            "every unique HR-selected deployable adaptation must leave frontier "
            ">=5% better in both Finance mean completion pages and unit-cost, "
            "with no adaptation Pareto dominance"
        ),
        "checks": checks,
        "decision": decision,
        "paper_action": (
            "retain strong beat-CaGR claim"
            if decision == "STRONG GO"
            else "downgrade/stop beat-CaGR claim and expose counterexample"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Phase 1: only these two HR BM25 files are opened.  Do not move Finance or
    # full-domain loading above the selection freeze.
    hr_access = load_access_graph(args.data_root / "hr")
    selection_rows, selection = _select_on_hr(hr_access)
    selected = selection["selected"]
    frozen_digest = selection["selected_config_sha256"]

    # Phase 2: configuration is immutable before qrels/visual and Finance load.
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
        "stage": "preregistered-strong-cagr-adaptation-stress-test",
        "gpu_used": False,
        "cost_scope": "unit build/reload cost; not wall clock",
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
        "evaluation": evaluation,
        "finance_gate": gate,
        "mandatory_interpretation": (
            "this stress test overrides the weaker faithful-theta gate for any "
            "paper claim that ReprForge beats CaGR-style deployment adaptations"
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
