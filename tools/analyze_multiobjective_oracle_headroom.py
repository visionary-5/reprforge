#!/usr/bin/env python3
"""Probe registered clairvoyant multi-objective scheduling headroom."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reprforge.time_aligned_quality import AXES, time_aligned_quality_metrics
from tools.analyze_cagr_bounded_wait import (
    MODELS,
    SEEDS,
    _aggregate,
    _json_digest,
    _replay,
)
from tools.analyze_time_aligned_quality import _aggregate_axis
from tools.analyze_windowed_arrivals import load_domain


BOUNDED_CONFIG = {
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


def _config_id(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _candidate_grid() -> list[dict[str, Any]]:
    weights = [
        (quality / 4, completion / 4, deadline / 4)
        for quality in range(5)
        for completion in range(5 - quality)
        for deadline in [4 - quality - completion]
    ]
    return [
        {
            "lambda_quality": quality,
            "lambda_completion": completion,
            "lambda_deadline": deadline,
            "deadline_scale": deadline_scale,
            "future_wait_budget": wait_budget,
        }
        for quality, completion, deadline in weights
        for deadline_scale in (64, 256)
        for wait_budget in (0, 16)
    ]


def _endpoint_specs() -> dict[str, dict[str, Any]]:
    return {
        "fifo": {"policy": "fifo"},
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
        "frontier": {"policy": "frontier"},
        "bounded_cagr": {
            "policy": "cagr_faithful",
            "config": BOUNDED_CONFIG,
        },
    }


def _run_method(
    data: dict[str, Any],
    *,
    model: str,
    policy: str,
    config: dict[str, Any] | None,
) -> list[Any]:
    return [
        _replay(
            data,
            seed=seed,
            model=model,
            policy=policy,
            config=config,
            with_quality=True,
        )
        for seed in SEEDS
    ]


def _horizons(
    replay_sets: dict[str, list[Any]], seed_index: int
) -> dict[str, float]:
    return {
        axis: max(
            float(replays[seed_index].quality_publication_trace[-1][axis])
            for replays in replay_sets.values()
        )
        for axis in AXES
    }


def _summarize_method(
    data: dict[str, Any],
    replays: list[Any],
    horizons_by_seed: list[dict[str, float]],
) -> dict[str, Any]:
    axis_rows = {axis: [] for axis in AXES}
    runs = []
    for seed_index, (seed, replay) in enumerate(zip(SEEDS, replays)):
        trace = replay.quality_publication_trace
        axes = {
            axis: time_aligned_quality_metrics(
                trace,
                axis=axis,
                common_horizon=horizons_by_seed[seed_index][axis],
                base_quality=data["base_mean_quality"],
                final_quality=data["refined_mean_quality"],
            )
            for axis in AXES
        }
        for axis in AXES:
            axis_rows[axis].append(axes[axis])
        runs.append(
            {
                "seed": seed,
                "dispatch_order_sha256": _json_digest(replay.dispatch_order),
                "quality_publication_trace_sha256": _json_digest(trace),
                "axes": axes,
            }
        )
    system = _aggregate(replays, with_quality=True)
    return {
        "system": system,
        "axes": {
            axis: _aggregate_axis(axis_rows[axis]) for axis in AXES
        },
        "oracle_future_wait": {
            "events": sum(row.oracle_future_wait["events"] for row in replays),
            "total_unit_time": sum(
                row.oracle_future_wait["total_unit_time"] for row in replays
            ),
            "max_unit_time": max(
                row.oracle_future_wait["max_unit_time"] for row in replays
            ),
        },
        "runs": runs,
    }


def _evaluate_hr(data: dict[str, Any]) -> dict[str, Any]:
    configs = _candidate_grid()
    by_model = {}
    for model in MODELS:
        replay_sets = {
            name: _run_method(
                data,
                model=model,
                policy=spec["policy"],
                config=spec.get("config"),
            )
            for name, spec in _endpoint_specs().items()
        }
        for index, config in enumerate(configs):
            replay_sets[f"oracle_{index:02d}"] = _run_method(
                data,
                model=model,
                policy="multiobjective_oracle",
                config=config,
            )
        horizons_by_seed = [
            _horizons(replay_sets, seed_index) for seed_index in range(len(SEEDS))
        ]
        by_model[model] = {
            "common_horizons_by_seed": [
                {"seed": seed, **horizon}
                for seed, horizon in zip(SEEDS, horizons_by_seed)
            ],
            "endpoints": {
                name: _summarize_method(data, replay_sets[name], horizons_by_seed)
                for name in _endpoint_specs()
            },
            "oracles": {
                f"oracle_{index:02d}": _summarize_method(
                    data, replay_sets[f"oracle_{index:02d}"], horizons_by_seed
                )
                for index in range(len(configs))
            },
        }

    candidates = []
    for index, config in enumerate(configs):
        oracle_name = f"oracle_{index:02d}"
        ratios = []
        p95_ratios = []
        starvation = []
        safe = True
        model_rows = {}
        for model in MODELS:
            cell = by_model[model]
            oracle = cell["oracles"][oracle_name]
            bounded = cell["endpoints"]["bounded_cagr"]
            frontier = cell["endpoints"]["frontier"]
            p95_limit = min(
                bounded["system"]["sojourn_unit_time"]["p95"],
                frontier["system"]["sojourn_unit_time"]["p95"],
            )
            model_ratios = {
                "mean_sojourn_over_bounded": (
                    oracle["system"]["sojourn_unit_time"]["mean"]
                    / bounded["system"]["sojourn_unit_time"]["mean"]
                ),
                "work_per_query_over_bounded": (
                    oracle["system"]["unit_work_per_query"]
                    / bounded["system"]["unit_work_per_query"]
                ),
                "elapsed_regret_over_frontier": (
                    oracle["axes"]["elapsed_unit_time"][
                        "normalized_quality_regret_auc"
                    ]["mean"]
                    / frontier["axes"]["elapsed_unit_time"][
                        "normalized_quality_regret_auc"
                    ]["mean"]
                ),
            }
            p95_ratio = oracle["system"]["sojourn_unit_time"]["p95"] / p95_limit
            starvation_fraction = oracle["system"]["starvation"]["fraction"]
            parity = bool(
                oracle["system"]["dispatch_complete"]
                and oracle["system"]["final_union_pages"]
                == [data["candidate_union_pages"]]
            )
            safe = bool(
                safe
                and parity
                and p95_ratio <= 1.05
                and starvation_fraction <= 0.05
            )
            ratios.extend(model_ratios.values())
            p95_ratios.append(p95_ratio)
            starvation.append(starvation_fraction)
            model_rows[model] = {
                **model_ratios,
                "p95_over_best_endpoint": p95_ratio,
                "starvation_fraction": starvation_fraction,
                "parity": parity,
            }
        candidates.append(
            {
                "oracle_name": oracle_name,
                "config": config,
                "config_sha256": _json_digest(config),
                "safe": safe,
                "minimax_primary_ratio": max(ratios),
                "mean_primary_ratio": sum(ratios) / len(ratios),
                "max_p95_ratio": max(p95_ratios),
                "max_starvation_fraction": max(starvation),
                "by_arrival": model_rows,
            }
        )
    safe_rows = [row for row in candidates if row["safe"]]
    best_unconstrained = min(
        candidates,
        key=lambda row: (
            row["minimax_primary_ratio"],
            row["mean_primary_ratio"],
            row["max_p95_ratio"],
            row["max_starvation_fraction"],
            _config_id(row["config"]),
        ),
    )
    selected_row = (
        min(
            safe_rows,
            key=lambda row: (
                row["minimax_primary_ratio"],
                row["mean_primary_ratio"],
                row["max_p95_ratio"],
                row["max_starvation_fraction"],
                _config_id(row["config"]),
            ),
        )
        if safe_rows
        else None
    )
    selection = {
        "criterion": (
            "among HR safety-qualified candidates minimize the maximum of "
            "burst/Poisson sojourn-over-bounded, work-over-bounded, and "
            "elapsed-regret-over-frontier"
        ),
        "candidate_count": len(candidates),
        "safe_candidate_count": len(safe_rows),
        "p95_qualified_count": sum(
            all(
                value["p95_over_best_endpoint"] <= 1.05
                for value in row["by_arrival"].values()
            )
            for row in candidates
        ),
        "starvation_qualified_count": sum(
            row["max_starvation_fraction"] <= 0.05 for row in candidates
        ),
        "primary_endpoint_dominating_count": sum(
            all(
                max(
                    value["mean_sojourn_over_bounded"],
                    value["work_per_query_over_bounded"],
                    value["elapsed_regret_over_frontier"],
                )
                <= 1.0
                for value in row["by_arrival"].values()
            )
            for row in candidates
        ),
        "best_minimax_candidate_regardless_safety": {
            key: best_unconstrained[key]
            for key in (
                "oracle_name",
                "config",
                "config_sha256",
                "safe",
                "minimax_primary_ratio",
                "mean_primary_ratio",
                "max_p95_ratio",
                "max_starvation_fraction",
                "by_arrival",
            )
        },
        "selected_oracle_name": (
            None if selected_row is None else selected_row["oracle_name"]
        ),
        "selected": None if selected_row is None else selected_row["config"],
        "selected_config_sha256": _json_digest(
            None if selected_row is None else selected_row["config"]
        ),
        "finance_opened_during_selection": False,
        "candidate_table": candidates,
    }
    return {"selection": selection, "evaluation": by_model}


def _evaluate_finance(
    data: dict[str, Any], selected: dict[str, Any] | None
) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        replay_sets = {
            name: _run_method(
                data,
                model=model,
                policy=spec["policy"],
                config=spec.get("config"),
            )
            for name, spec in _endpoint_specs().items()
        }
        if selected is not None:
            replay_sets["hr_selected_oracle"] = _run_method(
                data,
                model=model,
                policy="multiobjective_oracle",
                config=selected,
            )
        horizons_by_seed = [
            _horizons(replay_sets, seed_index) for seed_index in range(len(SEEDS))
        ]
        by_model[model] = {
            "common_horizons_by_seed": [
                {"seed": seed, **horizon}
                for seed, horizon in zip(SEEDS, horizons_by_seed)
            ],
            "methods": {
                name: _summarize_method(data, replays, horizons_by_seed)
                for name, replays in replay_sets.items()
            },
        }
    return by_model


def _finance_gate(
    finance: dict[str, Any], selected: dict[str, Any] | None
) -> dict[str, Any]:
    if selected is None:
        return {
            "decision": "NO HEADROOM IN REGISTERED FAMILY",
            "no_safe_hr_selection": True,
            "checks": [],
        }
    checks = []
    for model in MODELS:
        methods = finance[model]["methods"]
        oracle = methods["hr_selected_oracle"]
        bounded = methods["bounded_cagr"]
        frontier = methods["frontier"]
        oracle_sojourn = oracle["system"]["sojourn_unit_time"]["mean"]
        bounded_sojourn = bounded["system"]["sojourn_unit_time"]["mean"]
        oracle_work = oracle["system"]["unit_work_per_query"]
        bounded_work = bounded["system"]["unit_work_per_query"]
        oracle_regret = oracle["axes"]["elapsed_unit_time"][
            "normalized_quality_regret_auc"
        ]["mean"]
        frontier_regret = frontier["axes"]["elapsed_unit_time"][
            "normalized_quality_regret_auc"
        ]["mean"]
        p95_limit = 1.05 * min(
            bounded["system"]["sojourn_unit_time"]["p95"],
            frontier["system"]["sojourn_unit_time"]["p95"],
        )
        improvements = {
            "mean_sojourn": 1.0 - oracle_sojourn / bounded_sojourn,
            "charged_work_per_query": 1.0 - oracle_work / bounded_work,
            "elapsed_quality_regret": 1.0 - oracle_regret / frontier_regret,
        }
        constraints = {
            "mean_sojourn_no_worse_than_bounded": oracle_sojourn <= bounded_sojourn,
            "work_no_worse_than_bounded": oracle_work <= bounded_work,
            "elapsed_regret_no_worse_than_frontier": oracle_regret
            <= frontier_regret,
            "p95_within_1.05_best_endpoint": oracle["system"][
                "sojourn_unit_time"
            ]["p95"]
            <= p95_limit,
            "starvation_at_most_5_percent": oracle["system"]["starvation"][
                "fraction"
            ]
            <= 0.05,
            "at_least_one_primary_improves_5_percent": max(
                improvements.values()
            )
            >= 0.05,
        }
        checks.append(
            {
                "arrival_model": model,
                "improvements": improvements,
                "constraints": constraints,
                "passes": all(constraints.values()),
            }
        )
    decision = (
        "HEADROOM GO"
        if all(check["passes"] for check in checks)
        else "NO HEADROOM IN REGISTERED FAMILY"
    )
    return {
        "decision": decision,
        "no_safe_hr_selection": False,
        "checks": checks,
        "scope": (
            "finite registered greedy clairvoyant family only; not a global "
            "optimality or impossibility claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bounded-reference", type=Path, required=True)
    parser.add_argument("--time-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bounded_reference = json.loads(args.bounded_reference.read_text())
    time_reference = json.loads(args.time_reference.read_text())
    if _json_digest(BOUNDED_CONFIG) != bounded_reference["selection"][
        "selected_config_sha256"
    ]:
        raise ValueError("bounded endpoint config differs from frozen reference")

    hr = load_domain(args.data_root / "hr", 20)
    hr_result = _evaluate_hr(hr)
    selected = hr_result["selection"]["selected"]
    frozen_digest = hr_result["selection"]["selected_config_sha256"]

    if _json_digest(selected) != frozen_digest:
        raise AssertionError("oracle selection changed before Finance unseal")
    finance = load_domain(args.data_root / "finance", 20)
    if _json_digest(selected) != frozen_digest:
        raise AssertionError("oracle selection changed after Finance unseal")
    finance_result = _evaluate_finance(finance, selected)
    gate = _finance_gate(finance_result, selected)

    method_mapping = {
        "fifo": "fifo",
        "faithful_theta_0.5": "faithful_theta_0.5",
        "frontier": "frontier",
        "bounded_cagr": "hr_selected_bounded_cagr",
    }
    reference_audit = {}
    for domain, current in (
        ("hr", hr_result["evaluation"]),
        ("finance", finance_result),
    ):
        reference_audit[domain] = {}
        for model in MODELS:
            current_methods = (
                current[model]["endpoints"]
                if domain == "hr"
                else current[model]["methods"]
            )
            reference_audit[domain][model] = {
                method: {
                    "system_aggregate_exact_match": (
                        current_methods[method]["system"]
                        == time_reference["evaluation"][domain][model]["methods"][
                            reference_name
                        ]["system"]
                    ),
                    "order_sha256_exact_match": (
                        current_methods[method]["system"]["order_sha256"]
                        == time_reference["evaluation"][domain][model]["methods"][
                            reference_name
                        ]["system"]["order_sha256"]
                    ),
                }
                for method, reference_name in method_mapping.items()
            }

    report = {
        "schema_version": 1,
        "stage": "registered-multiobjective-clairvoyant-headroom-probe",
        "contract_commit": "69add9b",
        "gpu_used": False,
        "oracle_scope": (
            "finite 60-point greedy clairvoyant family; not mathematical optimum"
        ),
        "candidate_grid": {
            "count": len(_candidate_grid()),
            "weight_simplex_step": 0.25,
            "deadline_scales": [64, 256],
            "future_wait_budgets": [0, 16],
            "cost_denominator_epsilon": 1,
        },
        "references": {
            "bounded": {
                "path": str(args.bounded_reference.resolve()),
                "sha256": _sha256(args.bounded_reference),
            },
            "time_aligned": {
                "path": str(args.time_reference.resolve()),
                "sha256": _sha256(args.time_reference),
            },
        },
        "inputs": {
            "hr": {
                "query_count": hr["query_count"],
                "corpus_pages": hr["corpus_pages"],
                "candidate_union_pages": hr["candidate_union_pages"],
                "provenance": hr["provenance"],
            },
            "finance": {
                "query_count": finance["query_count"],
                "corpus_pages": finance["corpus_pages"],
                "candidate_union_pages": finance["candidate_union_pages"],
                "provenance": finance["provenance"],
            },
        },
        "hr": hr_result,
        "unseal_audit": {
            "selected_config_sha256_before_finance": frozen_digest,
            "selected_config_sha256_after_finance": _json_digest(selected),
            "finance_loaded_after_hr_selection": True,
        },
        "finance": finance_result,
        "finance_gate": gate,
        "endpoint_reference_audit": reference_audit,
        "all_endpoint_system_aggregates_match_time_reference": all(
            row["system_aggregate_exact_match"]
            for domain in reference_audit.values()
            for model in domain.values()
            for row in model.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selection": {
                    key: value
                    for key, value in hr_result["selection"].items()
                    if key != "candidate_table"
                },
                "finance_gate": gate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
