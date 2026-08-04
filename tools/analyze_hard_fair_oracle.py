#!/usr/bin/env python3
"""Evaluate a finite hard younger-bypass constrained oracle family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.analyze_cagr_bounded_wait import MODELS, SEEDS, _json_digest
from tools.analyze_multiobjective_oracle_headroom import (
    _endpoint_specs,
    _horizons,
    _run_method,
    _sha256,
    _summarize_method,
)
from tools.analyze_windowed_arrivals import load_domain


BASE_ORACLE_CONFIG = {
    "lambda_quality": 0.0,
    "lambda_completion": 0.75,
    "lambda_deadline": 0.25,
    "deadline_scale": 256,
    "future_wait_budget": 16,
}
BYPASS_BUDGETS = (8, 16, 32, 64)


def _config_id(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _candidate_grid() -> list[dict[str, Any]]:
    return [
        {**BASE_ORACLE_CONFIG, "bypass_budget": budget}
        for budget in BYPASS_BUDGETS
    ]


def _hard_fair_aggregate(replays: list[Any]) -> dict[str, Any]:
    selections = sum(
        row.oracle_hard_fairness["selection_count"] for row in replays
    )
    forced = sum(
        row.oracle_hard_fairness["forced_selection_count"] for row in replays
    )
    return {
        "runs": len(replays),
        "configured_bypass_budgets": sorted(
            {
                row.oracle_hard_fairness["configured_bypass_budget"]
                for row in replays
            }
        ),
        "selection_count": selections,
        "forced_selection_count": forced,
        "forced_selection_fraction": forced / selections if selections else 0.0,
        "protected_unique_queries_sum": sum(
            row.oracle_hard_fairness["protected_unique_queries"] for row in replays
        ),
        "mean_protected_query_fraction": sum(
            row.oracle_hard_fairness["protected_query_fraction"] for row in replays
        )
        / len(replays),
        "max_final_younger_bypass": max(
            row.oracle_hard_fairness["max_final_younger_bypass"]
            for row in replays
        ),
        "budget_violation_count": sum(
            row.oracle_hard_fairness["budget_violation_count"] for row in replays
        ),
    }


def _summarize_hard_method(
    data: dict[str, Any],
    replays: list[Any],
    horizons_by_seed: list[dict[str, float]],
) -> dict[str, Any]:
    result = _summarize_method(data, replays, horizons_by_seed)
    result["hard_fairness"] = _hard_fair_aggregate(replays)
    return result


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = left["pareto_vector"]
    right_values = right["pareto_vector"]
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _select_knee(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [row for row in candidates if row["qualified"]]
    pareto = [
        row
        for row in qualified
        if not any(_dominates(other, row) for other in qualified if other is not row)
    ]
    if not pareto:
        return {
            "qualified_count": len(qualified),
            "pareto_count": 0,
            "pareto_configs": [],
            "selected": None,
            "selected_config_sha256": _json_digest(None),
        }
    dimensions = len(pareto[0]["pareto_vector"])
    minima = [min(row["pareto_vector"][i] for row in pareto) for i in range(dimensions)]
    maxima = [max(row["pareto_vector"][i] for row in pareto) for i in range(dimensions)]
    for row in pareto:
        normalized = [
            (row["pareto_vector"][i] - minima[i]) / (maxima[i] - minima[i])
            if maxima[i] > minima[i]
            else 0.0
            for i in range(dimensions)
        ]
        row["knee_normalized_vector"] = normalized
        row["knee_distance"] = sum(value * value for value in normalized) ** 0.5
    selected_row = min(
        pareto,
        key=lambda row: (
            row["knee_distance"],
            row["max_primary_ratio"],
            row["mean_primary_ratio"],
            row["config"]["bypass_budget"],
            _config_id(row["config"]),
        ),
    )
    return {
        "qualified_count": len(qualified),
        "pareto_count": len(pareto),
        "pareto_configs": [row["oracle_name"] for row in pareto],
        "normalization_minima": minima,
        "normalization_maxima": maxima,
        "selected_oracle_name": selected_row["oracle_name"],
        "selected": selected_row["config"],
        "selected_config_sha256": selected_row["config_sha256"],
        "selected_knee_distance": selected_row["knee_distance"],
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
            replay_sets[f"hard_oracle_{index}"] = _run_method(
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
                f"hard_oracle_{index}": _summarize_hard_method(
                    data, replay_sets[f"hard_oracle_{index}"], horizons_by_seed
                )
                for index in range(len(configs))
            },
        }

    candidates = []
    for index, config in enumerate(configs):
        oracle_name = f"hard_oracle_{index}"
        ratios = []
        qualified = True
        arrival_rows = {}
        actual_max_bypass = 0
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
            starvation = oracle["system"]["starvation"]["fraction"]
            violations = oracle["hard_fairness"]["budget_violation_count"]
            parity = bool(
                oracle["system"]["dispatch_complete"]
                and oracle["system"]["final_union_pages"]
                == [data["candidate_union_pages"]]
            )
            model_qualified = bool(
                parity
                and violations == 0
                and starvation <= 0.05
                and p95_ratio <= 1.05
                and all(value <= 1.0 for value in model_ratios.values())
            )
            qualified = qualified and model_qualified
            ratios.extend(model_ratios.values())
            actual_max_bypass = max(
                actual_max_bypass,
                oracle["hard_fairness"]["max_final_younger_bypass"],
            )
            arrival_rows[model] = {
                **model_ratios,
                "p95_over_best_endpoint": p95_ratio,
                "starvation_fraction": starvation,
                "budget_violation_count": violations,
                "actual_max_younger_bypass": oracle["hard_fairness"][
                    "max_final_younger_bypass"
                ],
                "forced_selection_fraction": oracle["hard_fairness"][
                    "forced_selection_fraction"
                ],
                "qualified": model_qualified,
                "parity": parity,
            }
        candidates.append(
            {
                "oracle_name": oracle_name,
                "config": config,
                "config_sha256": _json_digest(config),
                "qualified": qualified,
                "max_primary_ratio": max(ratios),
                "mean_primary_ratio": sum(ratios) / len(ratios),
                "actual_max_younger_bypass": actual_max_bypass,
                "pareto_vector": [*ratios, actual_max_bypass / 64.0],
                "by_arrival": arrival_rows,
            }
        )
    selection = _select_knee(candidates)
    selection.update(
        {
            "criterion": (
                "seven-axis qualified Pareto set followed by min-max normalized "
                "equal-weight distance-to-ideal knee"
            ),
            "candidate_count": len(candidates),
            "finance_data_used_during_selection": False,
            "candidate_table": candidates,
        }
    )
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
            replay_sets["hr_selected_hard_oracle"] = _run_method(
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
                name: (
                    _summarize_hard_method(data, replays, horizons_by_seed)
                    if name == "hr_selected_hard_oracle"
                    else _summarize_method(data, replays, horizons_by_seed)
                )
                for name, replays in replay_sets.items()
            },
        }
    return by_model


def _evaluate_causal_timeout_diagnostic(
    data: dict[str, Any], selected: dict[str, Any] | None
) -> dict[str, Any]:
    """Post-hoc online-timer audit; excluded from registered selection/gates."""

    if selected is None:
        return {
            "status": "not_run_without_hr_selection",
            "excluded_from_preregistered_selection_and_gate": True,
        }
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
        replay_sets["registered_end_aware"] = _run_method(
            data,
            model=model,
            policy="multiobjective_oracle",
            config=selected,
        )
        replay_sets["online_timer_no_end_signal"] = _run_method(
            data,
            model=model,
            policy="multiobjective_oracle",
            config={**selected, "wait_through_stream_end": True},
        )
        horizons_by_seed = [
            _horizons(replay_sets, seed_index) for seed_index in range(len(SEEDS))
        ]
        registered = replay_sets["registered_end_aware"]
        timer_only = replay_sets["online_timer_no_end_signal"]
        by_model[model] = {
            "common_horizons_by_seed": [
                {"seed": seed, **horizon}
                for seed, horizon in zip(SEEDS, horizons_by_seed)
            ],
            "methods": {
                name: (
                    _summarize_hard_method(data, replays, horizons_by_seed)
                    if name
                    in {"registered_end_aware", "online_timer_no_end_signal"}
                    else _summarize_method(data, replays, horizons_by_seed)
                )
                for name, replays in replay_sets.items()
            },
            "trace_comparison_by_seed": [
                {
                    "seed": seed,
                    "dispatch_order_exact_match": (
                        nominal.dispatch_order == causal.dispatch_order
                    ),
                    "charged_work_exact_match": (
                        nominal.total_unit_work == causal.total_unit_work
                    ),
                    "final_union_exact_match": (
                        nominal.final_unique_pages == causal.final_unique_pages
                    ),
                    "elapsed_finish_delta_unit_time": (
                        causal.quality_publication_trace[-1]["elapsed_unit_time"]
                        - nominal.quality_publication_trace[-1]["elapsed_unit_time"]
                    ),
                    "registered_wait_unit_time": nominal.oracle_future_wait[
                        "total_unit_time"
                    ],
                    "online_timer_wait_unit_time": causal.oracle_future_wait[
                        "total_unit_time"
                    ],
                }
                for seed, nominal, causal in zip(SEEDS, registered, timer_only)
            ],
        }
    all_rows = [
        row
        for model in by_model.values()
        for row in model["trace_comparison_by_seed"]
    ]
    return {
        "status": "completed",
        "excluded_from_preregistered_selection_and_gate": True,
        "purpose": (
            "replace the finite replay's explicit end-of-stream flush with the "
            "same online timeout used before stream end"
        ),
        "selected_dispatch_uses_qrel_gain": False,
        "selected_dispatch_state": [
            "arrived locator cohort membership",
            "current compiled-page set",
            "current active LRU state",
            "arrival rank and current age",
            "online younger-bypass counters",
        ],
        "wait_observability": {
            "before_stream_end": (
                "timer and observed arrival events only; no future arrival time or "
                "content is inspected"
            ),
            "registered_finite_tail": (
                "flushes immediately after the replay knows there are no more arrivals; "
                "this needs an explicit end-of-stream/session signal"
            ),
            "conservative_online_variant": (
                "without an end signal, waits until the already-armed oldest-query "
                "deadline"
            ),
        },
        "all_dispatch_orders_exact_match": all(
            row["dispatch_order_exact_match"] for row in all_rows
        ),
        "all_charged_work_exact_match": all(
            row["charged_work_exact_match"] for row in all_rows
        ),
        "all_final_unions_exact_match": all(
            row["final_union_exact_match"] for row in all_rows
        ),
        "evaluation": by_model,
    }


def _finance_gate(
    finance: dict[str, Any],
    selected: dict[str, Any] | None,
    *,
    expected_union: int,
    oracle_method: str = "hr_selected_hard_oracle",
) -> dict[str, Any]:
    if selected is None:
        return {
            "decision": "NO HEADROOM IN REGISTERED HARD-FAIR FAMILY",
            "no_qualified_hr_selection": True,
            "checks": [],
        }
    checks = []
    for model in MODELS:
        methods = finance[model]["methods"]
        oracle = methods[oracle_method]
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
        improvements = {
            "mean_sojourn": 1.0 - oracle_sojourn / bounded_sojourn,
            "charged_work_per_query": 1.0 - oracle_work / bounded_work,
            "elapsed_quality_regret": 1.0 - oracle_regret / frontier_regret,
        }
        p95_limit = 1.05 * min(
            bounded["system"]["sojourn_unit_time"]["p95"],
            frontier["system"]["sojourn_unit_time"]["p95"],
        )
        constraints = {
            "parity": bool(
                oracle["system"]["dispatch_complete"]
                and oracle["system"]["final_union_pages"] == [expected_union]
            ),
            "hard_budget_violation_zero": oracle["hard_fairness"][
                "budget_violation_count"
            ]
            == 0,
            "starvation_at_most_5_percent": oracle["system"]["starvation"][
                "fraction"
            ]
            <= 0.05,
            "p95_within_1.05_best_endpoint": oracle["system"][
                "sojourn_unit_time"
            ]["p95"]
            <= p95_limit,
            "mean_sojourn_no_worse_than_bounded": oracle_sojourn <= bounded_sojourn,
            "work_no_worse_than_bounded": oracle_work <= bounded_work,
            "elapsed_regret_no_worse_than_frontier": oracle_regret
            <= frontier_regret,
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
    return {
        "decision": (
            "HARD-FAIR HEADROOM GO"
            if all(check["passes"] for check in checks)
            else "NO HEADROOM IN REGISTERED HARD-FAIR FAMILY"
        ),
        "no_qualified_hr_selection": False,
        "checks": checks,
        "scope": "finite constrained oracle family; not a deployable method",
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_selection_first_inputs(
    data_root: Path,
    time_reference_path: Path,
    *,
    domain_loader: Any = load_domain,
    hr_evaluator: Any = _evaluate_hr,
    reference_loader: Any = _read_json,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    str,
    dict[str, Any],
    dict[str, Any],
]:
    """Freeze HR selection before reading any Finance-bearing input."""

    hr = domain_loader(data_root / "hr", 20)
    hr_result = hr_evaluator(hr)
    selected = hr_result["selection"]["selected"]
    frozen_digest = hr_result["selection"]["selected_config_sha256"]
    if _json_digest(selected) != frozen_digest:
        raise AssertionError("hard-fair selection digest was not frozen on HR")

    # The time reference contains a Finance evaluation, so it belongs on the
    # transfer side of the boundary even though it is used only for endpoint
    # parity auditing.
    time_reference = reference_loader(time_reference_path)
    finance = domain_loader(data_root / "finance", 20)
    return hr, hr_result, selected, frozen_digest, time_reference, finance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--time-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    (
        hr,
        hr_result,
        selected,
        frozen_digest,
        time_reference,
        finance,
    ) = _load_selection_first_inputs(args.data_root, args.time_reference)
    if _json_digest(selected) != frozen_digest:
        raise AssertionError("hard-fair selection changed during frozen transfer")
    finance_result = _evaluate_finance(finance, selected)
    gate = _finance_gate(
        finance_result,
        selected,
        expected_union=finance["candidate_union_pages"],
    )
    observability_audit = _evaluate_causal_timeout_diagnostic(finance, selected)
    if selected is not None:
        observability_audit["posthoc_online_timer_gate"] = _finance_gate(
            observability_audit["evaluation"],
            selected,
            expected_union=finance["candidate_union_pages"],
            oracle_method="online_timer_no_end_signal",
        )

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
            methods = (
                current[model]["endpoints"]
                if domain == "hr"
                else current[model]["methods"]
            )
            reference_audit[domain][model] = {
                method: {
                    "system_aggregate_exact_match": (
                        methods[method]["system"]
                        == time_reference["evaluation"][domain][model]["methods"][
                            reference_name
                        ]["system"]
                    ),
                    "order_sha256_exact_match": (
                        methods[method]["system"]["order_sha256"]
                        == time_reference["evaluation"][domain][model]["methods"][
                            reference_name
                        ]["system"]["order_sha256"]
                    ),
                }
                for method, reference_name in method_mapping.items()
            }

    report = {
        "schema_version": 1,
        "stage": "registered-hard-fair-clairvoyant-headroom-probe",
        "contract_commit": "060d131",
        "gpu_used": False,
        "scope": "finite constrained oracle family; not a deployable method",
        "fixed_utility_config": BASE_ORACLE_CONFIG,
        "candidate_bypass_budgets": list(BYPASS_BUDGETS),
        "time_reference": {
            "path": str(args.time_reference.resolve()),
            "sha256": _sha256(args.time_reference),
        },
        "inputs": {
            domain: {
                "query_count": data["query_count"],
                "corpus_pages": data["corpus_pages"],
                "candidate_union_pages": data["candidate_union_pages"],
                "provenance": data["provenance"],
            }
            for domain, data in (("hr", hr), ("finance", finance))
        },
        "hr": hr_result,
        "frozen_transfer_provenance_audit": {
            "selected_config_sha256_before_transfer_reads": frozen_digest,
            "selected_config_sha256_after_transfer_evaluation": _json_digest(
                selected
            ),
            "configuration_selection_has_finance_data_dependency": False,
            "code_path_selection_first": True,
            "finance_bearing_reference_loaded_after_hr_selection": True,
            "finance_workload_loaded_after_hr_selection": True,
            "finance_tuning_performed": False,
            "historical_finance_endpoints_already_existed": True,
            "human_level_novel_blind_test_claimed": False,
        },
        "finance": finance_result,
        "finance_gate": gate,
        "selected_policy_observability_audit": observability_audit,
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
                "endpoint_reference_match": report[
                    "all_endpoint_system_aggregates_match_time_reference"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
