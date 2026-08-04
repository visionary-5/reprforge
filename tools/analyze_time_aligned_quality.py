#!/usr/bin/env python3
"""Measure post-hoc anytime quality on three explicitly separated axes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.time_aligned_quality import AXES, time_aligned_quality_metrics
from tools.analyze_cagr_bounded_wait import (
    MODELS,
    SEEDS,
    _aggregate,
    _json_digest,
    _replay,
)
from tools.analyze_windowed_arrivals import load_domain


SELECTED_BOUNDED_CAGR = {
    "capacity": 80,
    "cross_group_fill": True,
    "family": "fixed_jaccard",
    "group_pool": 64,
    "min_pending": 4,
    "target_group_size": 16,
    "wait_budget": 16,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _method_specs() -> dict[str, dict[str, Any]]:
    return {
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
        "hr_selected_bounded_cagr": {
            "policy": "cagr_faithful",
            "config": SELECTED_BOUNDED_CAGR,
        },
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _optional_distribution(values: list[float | None]) -> dict[str, float] | None:
    if any(value is None for value in values):
        return None
    return _distribution([float(value) for value in values])


def _aggregate_axis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runs": len(rows),
        "common_horizon": _distribution(
            [float(row["common_horizon"]) for row in rows]
        ),
        "method_endpoint": _distribution(
            [float(row["method_endpoint"]) for row in rows]
        ),
        "method_endpoint_fraction": _distribution(
            [float(row["method_endpoint_fraction"]) for row in rows]
        ),
        "mean_quality_auc": _distribution(
            [float(row["mean_quality_auc"]) for row in rows]
        ),
        "raw_signed_quality_gain_auc": _distribution(
            [float(row["raw_signed_quality_gain_auc"]) for row in rows]
        ),
        "normalized_quality_gain_auc": _optional_distribution(
            [row["normalized_quality_gain_auc"] for row in rows]
        ),
        "normalized_quality_regret_auc": _optional_distribution(
            [row["normalized_quality_regret_auc"] for row in rows]
        ),
        "attainment": {},
        "fixed_budgets": {},
    }
    for key in rows[0]["attainment"]:
        values = [row["attainment"][key] for row in rows]
        if any(value is None for value in values):
            result["attainment"][key] = None
        else:
            result["attainment"][key] = {
                "coordinate": _distribution(
                    [float(value["coordinate"]) for value in values]
                ),
                "common_horizon_fraction": _distribution(
                    [float(value["common_horizon_fraction"]) for value in values]
                ),
            }
    for key in rows[0]["fixed_budgets"]:
        values = [row["fixed_budgets"][key] for row in rows]
        result["fixed_budgets"][key] = {
            "coordinate": _distribution(
                [float(value["coordinate"]) for value in values]
            ),
            "mean_quality": _distribution(
                [float(value["mean_quality"]) for value in values]
            ),
            "quality_gain_over_base": _distribution(
                [float(value["quality_gain_over_base"]) for value in values]
            ),
            "final_positive_gain_fraction_achieved": _optional_distribution(
                [value["final_positive_gain_fraction_achieved"] for value in values]
            ),
        }
    return result


def _run_cell(data: dict[str, Any], model: str) -> dict[str, Any]:
    specs = _method_specs()
    quality_replays: dict[str, list[Any]] = {name: [] for name in specs}
    zero_gain_order_match: dict[str, list[bool]] = {name: [] for name in specs}
    for seed in SEEDS:
        for name, spec in specs.items():
            quality = _replay(
                data,
                seed=seed,
                model=model,
                policy=spec["policy"],
                config=spec.get("config"),
                with_quality=True,
            )
            zero_gain = _replay(
                data,
                seed=seed,
                model=model,
                policy=spec["policy"],
                config=spec.get("config"),
                with_quality=False,
            )
            quality_replays[name].append(quality)
            zero_gain_order_match[name].append(
                quality.dispatch_order == zero_gain.dispatch_order
            )

    horizons_by_seed = []
    run_metrics: dict[str, list[dict[str, Any]]] = {name: [] for name in specs}
    axis_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {axis: [] for axis in AXES} for name in specs
    }
    for seed_index, seed in enumerate(SEEDS):
        horizons = {
            axis: max(
                float(
                    quality_replays[name][seed_index]
                    .quality_publication_trace[-1][axis]
                )
                for name in specs
            )
            for axis in AXES
        }
        horizons_by_seed.append({"seed": seed, **horizons})
        for name in specs:
            replay = quality_replays[name][seed_index]
            trace = replay.quality_publication_trace
            if float(trace[-1]["charged_unit_work"]) != replay.total_unit_work:
                raise AssertionError("trace charged-work endpoint differs from replay")
            if int(trace[-1]["unique_compiled_pages"]) != replay.final_unique_pages:
                raise AssertionError("trace unique-page endpoint differs from replay")
            if int(trace[-1]["published_queries"]) != data["query_count"]:
                raise AssertionError("trace does not publish every query")
            metrics = {
                axis: time_aligned_quality_metrics(
                    trace,
                    axis=axis,
                    common_horizon=horizons[axis],
                    base_quality=data["base_mean_quality"],
                    final_quality=data["refined_mean_quality"],
                )
                for axis in AXES
            }
            for axis in AXES:
                axis_rows[name][axis].append(metrics[axis])
            run_metrics[name].append(
                {
                    "seed": seed,
                    "dispatch_order_sha256": _json_digest(replay.dispatch_order),
                    "quality_publication_trace_sha256": _json_digest(trace),
                    "publication_batches": len(trace) - 1,
                    "qrel_free_dispatch_order_match": zero_gain_order_match[name][
                        seed_index
                    ],
                    "axes": metrics,
                }
            )

    methods = {}
    for name, spec in specs.items():
        system = _aggregate(quality_replays[name], with_quality=True)
        methods[name] = {
            "policy": spec["policy"],
            "config": spec.get("config"),
            "system": system,
            "axes": {
                axis: _aggregate_axis(axis_rows[name][axis]) for axis in AXES
            },
            "legacy_metric_names": {
                "quality_work_auc": {
                    "value": system["quality_work_auc"],
                    "actual_semantics": (
                        "quality integrated on unique compiled pages then extended "
                        "to corpus_pages; not elapsed time or charged work"
                    ),
                },
                "normalized_quality_regret_auc": {
                    "value": system["normalized_quality_regret_auc"],
                    "actual_semantics": "normalized regret on unique compiled pages",
                },
            },
            "all_qrel_free_dispatch_orders_match": all(
                zero_gain_order_match[name]
            ),
            "runs": run_metrics[name],
        }
    return {"common_horizons_by_seed": horizons_by_seed, "methods": methods}


def _finance_interpretation(finance: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for model in MODELS:
        methods = finance[model]["methods"]
        adaptation = methods["hr_selected_bounded_cagr"]
        frontier = methods["frontier"]
        adaptation_system = adaptation["system"]
        frontier_system = frontier["system"]
        adaptation_regret = adaptation["axes"]["elapsed_unit_time"][
            "normalized_quality_regret_auc"
        ]["mean"]
        frontier_regret = frontier["axes"]["elapsed_unit_time"][
            "normalized_quality_regret_auc"
        ]["mean"]
        system_dominance = bool(
            adaptation_system["sojourn_unit_time"]["mean"]
            <= frontier_system["sojourn_unit_time"]["mean"]
            and adaptation_system["unit_work_per_query"]
            <= frontier_system["unit_work_per_query"]
        )
        three_axis_dominance = bool(
            system_dominance
            and adaptation_regret <= frontier_regret
            and (
                adaptation_system["sojourn_unit_time"]["mean"]
                < frontier_system["sojourn_unit_time"]["mean"]
                or adaptation_system["unit_work_per_query"]
                < frontier_system["unit_work_per_query"]
                or adaptation_regret < frontier_regret
            )
        )
        tradeoff = bool(system_dominance and frontier_regret < adaptation_regret)
        checks.append(
            {
                "arrival_model": model,
                "adaptation_system_axis_dominance": system_dominance,
                "adaptation_elapsed_quality_regret": adaptation_regret,
                "frontier_elapsed_quality_regret": frontier_regret,
                "frontier_regret_reduction_vs_adaptation": (
                    1.0 - frontier_regret / adaptation_regret
                    if adaptation_regret
                    else None
                ),
                "adaptation_three_axis_dominance": three_axis_dominance,
                "system_quality_tradeoff": tradeoff,
            }
        )
    if all(check["adaptation_three_axis_dominance"] for check in checks):
        decision = "THREE-AXIS DOMINANCE"
    elif any(check["system_quality_tradeoff"] for check in checks):
        decision = "SYSTEM-QUALITY TRADEOFF"
    else:
        decision = "MIXED/INCONCLUSIVE"
    return {
        "primary_axes": [
            "mean arrival-to-publication sojourn",
            "charged unit work per query",
            "elapsed-unit-time normalized quality regret AUC",
        ],
        "checks": checks,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bounded-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bounded = json.loads(args.bounded_reference.read_text())
    selected_hash = _json_digest(SELECTED_BOUNDED_CAGR)
    if selected_hash != bounded["selection"]["selected_config_sha256"]:
        raise ValueError("frozen bounded CaGR configuration digest differs")

    domains = {
        "hr": load_domain(args.data_root / "hr", 20),
        "finance": load_domain(args.data_root / "finance", 20),
    }
    evaluation = {
        domain: {model: _run_cell(data, model) for model in MODELS}
        for domain, data in domains.items()
    }

    reference_names = {"hr": "hr_post_hoc", "finance": "finance_sealed"}
    reference_audit = {}
    for domain in domains:
        reference_audit[domain] = {}
        for model in MODELS:
            reference_audit[domain][model] = {}
            for method, value in evaluation[domain][model]["methods"].items():
                reference = bounded["evaluation"][reference_names[domain]][model][
                    method
                ]["aggregate"]
                reference_audit[domain][model][method] = {
                    "system_aggregate_exact_match": value["system"] == reference,
                    "order_sha256_exact_match": (
                        value["system"]["order_sha256"]
                        == reference["order_sha256"]
                    ),
                }

    report = {
        "schema_version": 1,
        "stage": "post-hoc-time-aligned-anytime-quality",
        "contract_commit": "a39e7ca",
        "gpu_used": False,
        "selected_bounded_cagr": SELECTED_BOUNDED_CAGR,
        "selected_config_sha256": selected_hash,
        "metric_contract": {
            "quality_state": (
                "unpublished query uses BM25 nDCG@10; atomically published query "
                "uses frozen Top-20 z-score fusion nDCG@10"
            ),
            "axes": {
                "elapsed_unit_time": (
                    "build/reload/prefetch=1, hit=0, including idle and explicit wait"
                ),
                "charged_unit_work": (
                    "demand/prefetch build or reload=1; hit/idle/wait=0"
                ),
                "unique_compiled_pages": "distinct first-build pages only",
            },
            "shared_horizon": (
                "per domain/arrival/seed/axis maximum endpoint over all methods"
            ),
            "qrel_use": "post-hoc metric only; verified against zero-gain dispatch",
            "quality_curves_monotonized_or_clipped": False,
        },
        "bounded_reference": {
            "path": str(args.bounded_reference.resolve()),
            "sha256": _sha256(args.bounded_reference),
        },
        "inputs": {
            domain: {
                "query_count": data["query_count"],
                "corpus_pages": data["corpus_pages"],
                "candidate_union_pages": data["candidate_union_pages"],
                "base_mean_quality": data["base_mean_quality"],
                "refined_mean_quality": data["refined_mean_quality"],
                "provenance": data["provenance"],
            }
            for domain, data in domains.items()
        },
        "evaluation": evaluation,
        "reference_audit": reference_audit,
        "all_system_aggregates_match_bounded_reference": all(
            row["system_aggregate_exact_match"]
            for domain in reference_audit.values()
            for model in domain.values()
            for row in model.values()
        ),
        "all_qrel_free_dispatch_orders_match": all(
            method["all_qrel_free_dispatch_orders_match"]
            for domain in evaluation.values()
            for model in domain.values()
            for method in model["methods"].values()
        ),
        "finance_interpretation": _finance_interpretation(evaluation["finance"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "all_system_aggregates_match_bounded_reference": report[
                    "all_system_aggregates_match_bounded_reference"
                ],
                "all_qrel_free_dispatch_orders_match": report[
                    "all_qrel_free_dispatch_orders_match"
                ],
                "finance_interpretation": report["finance_interpretation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
