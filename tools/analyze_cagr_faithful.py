#!/usr/bin/env python3
"""Run the preregistered CaGR-RAG novelty gate on frozen HR/Finance traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.cagr_faithful_replay import POLICIES, replay_cagr_comparison
from reprforge.windowed_arrival_replay import make_arrival_times
from tools.analyze_windowed_arrivals import load_domain


SEEDS = (20260804, 20260805, 20260806, 20260807, 20260808)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _measured_profile(repo_root: Path, domain: str) -> dict[str, Any]:
    path = repo_root / "results" / "systems" / f"cohort-compiler-{domain}.json"
    if not path.is_file():
        return {"available": False, "reason": "cohort compiler artifact missing"}
    payload = json.loads(path.read_text())
    row = payload.get("online", {}).get("b8-resident")
    if not row:
        return {
            "available": False,
            "path": str(path),
            "reason": "frozen b8-resident measurement missing",
        }
    pages = int(row["visual_pages_encoded"])
    search_ms = float(row["search_ms"])
    return {
        "available": False,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "hardware": payload.get("hardware"),
        "observed_b8_search_ms": search_ms,
        "observed_pages_encoded": pages,
        "build_path_ms_per_page_diagnostic": search_ms / pages,
        "reason": (
            "artifact aggregates build, scoring, and request overhead and has no "
            "compiled-page reload measurement; a full measured-cost replay is "
            "therefore unavailable and cannot enter the gate"
        ),
    }


def _run_one(
    loaded: dict[str, Any],
    *,
    seed: int,
    arrival_model: str,
    policy: str,
    group_pool: int = 64,
    membership_rule: str = "max",
    cache_capacity: int = 80,
) -> tuple[Any, dict[str, Any]]:
    arrival_order = np.random.default_rng(seed).permutation(loaded["query_count"])
    model_offset = 0 if arrival_model == "burst" else 1
    arrival_times = make_arrival_times(
        loaded["query_count"],
        model=arrival_model,
        seed=seed + 10000 * model_offset,
        burst_size=32,
        burst_interval=64.0,
        poisson_mean=4.0,
    )
    replay = replay_cagr_comparison(
        loaded["cohorts"],
        arrival_order,
        arrival_times,
        loaded["quality_gain"],
        base_mean_quality=loaded["base_mean_quality"],
        corpus_pages=loaded["corpus_pages"],
        request_batch_size=8,
        window=64,
        policy=policy,
        cache_capacity=cache_capacity,
        cagr_group_pool=group_pool,
        cagr_theta=0.5,
        cagr_membership_rule=membership_rule,
    )
    summary = replay.as_dict(starvation_window=64)
    summary.update(
        {
            "seed": seed,
            "arrival_model": arrival_model,
            "dispatch_order_sha256": hashlib.sha256(
                np.asarray(replay.dispatch_order, dtype=np.int32).tobytes()
            ).hexdigest(),
        }
    )
    return replay, summary


def _aggregate(rows: list[tuple[Any, dict[str, Any]]]) -> dict[str, Any]:
    completion = np.concatenate(
        [np.asarray(replay.completion_pages, dtype=np.float64) for replay, _ in rows]
    )
    cost = np.concatenate(
        [
            np.asarray(replay.completion_unit_cost, dtype=np.float64)
            for replay, _ in rows
        ]
    )
    bypass = np.concatenate(
        [np.asarray(replay.bypass_count, dtype=np.int64) for replay, _ in rows]
    )
    auc = np.asarray([replay.quality_work_auc for replay, _ in rows])
    regret = np.asarray(
        [replay.normalized_quality_regret_auc for replay, _ in rows],
        dtype=np.float64,
    )
    demand_events = sum(replay.cache["demand_events"] for replay, _ in rows)
    cache_hits = sum(replay.cache["hits"] for replay, _ in rows)
    prefetch_events = sum(replay.prefetch["events"] for replay, _ in rows)
    useful = sum(replay.prefetch["useful"] for replay, _ in rows)
    group_counts = sum(replay.groups["count"] for replay, _ in rows)
    return {
        "runs": len(rows),
        "completion_pages": _distribution(completion),
        "completion_unit_cost": _distribution(cost),
        "quality_work_auc": _distribution(auc),
        "normalized_quality_regret_auc": _distribution(regret),
        "starvation": {
            "count": int(np.sum(bypass >= 64)),
            "fraction": float(np.mean(bypass >= 64)),
            "max_younger_bypass": int(bypass.max()),
        },
        "cache": {
            "demand_events": demand_events,
            "hits": cache_hits,
            "hit_fraction": cache_hits / demand_events if demand_events else 0.0,
            "builds": sum(replay.cache["builds"] for replay, _ in rows),
            "reloads": sum(replay.cache["reloads"] for replay, _ in rows),
        },
        "prefetch": {
            "events": prefetch_events,
            "useful": useful,
            "wasted": sum(replay.prefetch["wasted"] for replay, _ in rows),
            "precision": useful / prefetch_events if prefetch_events else None,
            "unused_unit_work": sum(
                replay.prefetch["unused_unit_work"] for replay, _ in rows
            ),
            "builds": sum(replay.prefetch["builds"] for replay, _ in rows),
            "reloads": sum(replay.prefetch["reloads"] for replay, _ in rows),
        },
        "groups": {
            "count": group_counts,
            "singleton_count": sum(
                replay.groups["singleton_count"] for replay, _ in rows
            ),
            "singleton_fraction": (
                sum(replay.groups["singleton_count"] for replay, _ in rows)
                / group_counts
                if group_counts
                else None
            ),
            "mean_of_run_mean_size": (
                float(
                    np.mean(
                        [
                            replay.groups["size_mean"]
                            for replay, _ in rows
                            if replay.groups["size_mean"] is not None
                        ]
                    )
                )
                if group_counts
                else None
            ),
            "max_size": (
                max(
                    replay.groups["size_max"]
                    for replay, _ in rows
                    if replay.groups["size_max"] is not None
                )
                if group_counts
                else None
            ),
        },
        "request_batches": {
            "count": sum(replay.request_batches["count"] for replay, _ in rows),
            "mean_query_slot_utilization": float(
                np.mean(
                    [
                        replay.request_batches["query_slots_used_fraction"]
                        for replay, _ in rows
                    ]
                )
            ),
        },
        "final_union_pages": sorted(
            {replay.final_unique_pages for replay, _ in rows}
        ),
    }


def _gate(datasets: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for domain in ("hr", "finance"):
        for model in ("burst", "poisson"):
            aggregate = datasets[domain]["aggregate"][model]
            cagr = aggregate["cagr_faithful"]
            frontier = aggregate["frontier"]
            cagr_mean = cagr["completion_pages"]["mean"]
            frontier_mean = frontier["completion_pages"]["mean"]
            improvement = 1.0 - frontier_mean / cagr_mean
            regret_ok = (
                frontier["normalized_quality_regret_auc"]["mean"]
                <= cagr["normalized_quality_regret_auc"]["mean"] + 1e-12
            )
            p95_page_ratio = (
                frontier["completion_pages"]["p95"]
                / cagr["completion_pages"]["p95"]
            )
            measured_available = bool(measured[domain]["available"])
            # Full measured cost is preregistered as unavailable if reload and
            # batch overhead cannot be identified independently.
            cost_improvement = None
            cost_p95_ratio = None
            passes = bool(improvement >= 0.05 and regret_ok and p95_page_ratio <= 1.05)
            checks.append(
                {
                    "domain": domain,
                    "arrival_model": model,
                    "frontier_vs_cagr_mean_completion_improvement": improvement,
                    "frontier_regret_no_worse": regret_ok,
                    "frontier_over_cagr_p95_page_ratio": p95_page_ratio,
                    "measured_cost_available": measured_available,
                    "measured_cost_improvement": cost_improvement,
                    "measured_cost_p95_ratio": cost_p95_ratio,
                    "qualifying_axis": "completion_page_work",
                    "passes": passes,
                }
            )
    return {
        "criterion": (
            "frontier improves mean completion page-work by >=5%, has no worse "
            "normalized quality regret, and P95 completion page-work <=1.05x "
            "CaGR-faithful in every HR/Finance x burst/Poisson setting"
        ),
        "checks": checks,
        "decision": "GO" if all(row["passes"] for row in checks) else "NO-GO",
        "failure_action": "STOP/retitle scheduler main claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    loaded = {
        domain: load_domain(args.data_root / domain, 20)
        for domain in ("hr", "finance")
    }
    datasets: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for domain, data in loaded.items():
        primary_rows: dict[str, dict[str, list[tuple[Any, dict[str, Any]]]]] = {
            model: {policy: [] for policy in POLICIES}
            for model in ("burst", "poisson")
        }
        run_summaries = []
        for seed in SEEDS:
            for model in ("burst", "poisson"):
                for policy in POLICIES:
                    replay, summary = _run_one(
                        data, seed=seed, arrival_model=model, policy=policy
                    )
                    primary_rows[model][policy].append((replay, summary))
                    run_summaries.append(summary)
        aggregate = {
            model: {
                policy: _aggregate(primary_rows[model][policy])
                for policy in POLICIES
            }
            for model in ("burst", "poisson")
        }
        datasets[domain] = {
            key: value
            for key, value in data.items()
            if key not in {"cohorts", "quality_gain"}
        }
        datasets[domain]["candidate_graph_sha256"] = _sha256_json(data["cohorts"])
        datasets[domain]["aggregate"] = aggregate
        datasets[domain]["runs"] = run_summaries

        domain_diagnostics = {}
        configurations = (
            ("pool20", 20, "max", 80),
            ("pool40", 40, "max", 80),
            ("equation3_all", 64, "all", 80),
            ("cache40", 64, "max", 40),
            ("cache160", 64, "max", 160),
        )
        for name, pool, rule, capacity in configurations:
            model_rows = {}
            for model in ("burst", "poisson"):
                rows = [
                    _run_one(
                        data,
                        seed=seed,
                        arrival_model=model,
                        policy="cagr_faithful",
                        group_pool=pool,
                        membership_rule=rule,
                        cache_capacity=capacity,
                    )
                    for seed in SEEDS
                ]
                model_rows[model] = _aggregate(rows)
            domain_diagnostics[name] = {
                "group_pool": pool,
                "membership_rule": rule,
                "cache_capacity": capacity,
                "aggregate": model_rows,
            }
        diagnostics[domain] = domain_diagnostics

    measured = {
        domain: _measured_profile(repo_root, domain)
        for domain in ("hr", "finance")
    }
    gate = _gate(datasets, measured)
    report = {
        "schema_version": 1,
        "stage": "preregistered-cagr-faithful-novelty-gate",
        "parameters": {
            "candidate_k": 20,
            "window": 64,
            "request_batch_max_queries": 8,
            "seeds": list(SEEDS),
            "arrival_models": {
                "burst": {"size": 32, "interval_page_work": 64.0},
                "poisson": {"mean_interarrival_page_work": 4.0},
            },
            "cagr": {
                "jaccard_theta": 0.5,
                "membership_rule": "max similarity to any member",
                "group_pool": 64,
                "group_order": "creation order",
                "within_group_order": "arrival order",
                "cross_group_batch_fill": False,
                "prefetch": "first query access set of next group",
            },
            "active_cache": {
                "policy": "deterministic LRU for every scheduler",
                "capacity_pages": 80,
                "derivation": "CaGR cache/nprobe ratio 40/10=4, times K20",
            },
        },
        "observation_contract": {
            "allowed": [
                "candidate-page sets of arrived queries",
                "arrival rank",
                "compiled pages and active-cache state",
                "past arrival candidate frequency for history popularity",
                "full-stream candidate frequency for the explicitly offline static-popularity diagnostic",
            ],
            "forbidden": [
                "future arrivals",
                "qrels",
                "visual scores",
                "quality gains or answer outcomes",
            ],
            "post_hoc_only": "qrels and visual scores compute frozen quality trajectory",
        },
        "measured_cost_profile": measured,
        "measured_cost_gate_note": (
            "full measured-cost replay is unavailable because frozen artifacts "
            "do not identify reload and request-overhead costs; gate uses page-work"
        ),
        "datasets": datasets,
        "sensitivity_diagnostics": diagnostics,
        "gate": gate,
        "fidelity_gaps": [
            "BM25 Top20 page access sets replace IVF nprobe10 cluster sets",
            "persistent visual-page construction differs from disk-vector loading",
            "W64 replaces CaGR's random 20-100 query batches; pools 20/40 are diagnostic",
            "atomic request batches contain at most 8 queries and never cross a CaGR group boundary",
            "no author official code repository was located as of 2026-08-04; implementation follows Algorithm 1",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
