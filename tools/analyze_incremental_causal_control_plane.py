#!/usr/bin/env python3
"""Verify exact incremental B32 semantics and deterministic operation cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.analyze_cagr_bounded_wait import MODELS, SEEDS, _json_digest
from tools.analyze_causal_hard_frontier import (
    TRANSFER_DOMAINS,
    _load_transfer_domains,
)
from tools.analyze_multiobjective_oracle_headroom import _run_method, _sha256


METHOD_SPECS = {
    "reference": "hard_budget_frontier",
    "transparent": "hard_budget_frontier_transparent",
    "incremental": "hard_budget_frontier_incremental",
    "delay_d32": "delay_scheduling_d32_control",
}


def _semantic_checks(reference: Any, candidate: Any) -> dict[str, bool]:
    return {
        "dispatch_order": reference.dispatch_order == candidate.dispatch_order,
        "completion_pages": reference.completion_pages == candidate.completion_pages,
        "completion_unit_time": (
            reference.completion_unit_cost == candidate.completion_unit_cost
        ),
        "wait_unit_time": reference.wait_unit_time == candidate.wait_unit_time,
        "sojourn_unit_time": reference.sojourn_unit_time == candidate.sojourn_unit_time,
        "bypass": reference.bypass_count == candidate.bypass_count,
        "final_union": reference.final_unique_pages == candidate.final_unique_pages,
        "total_work": reference.total_unit_work == candidate.total_unit_work,
        "cache": reference.cache == candidate.cache,
        "publication_trace": (
            reference.quality_publication_trace
            == candidate.quality_publication_trace
        ),
        "timer_wait": reference.oracle_future_wait == candidate.oracle_future_wait,
        "hard_fairness": (
            reference.oracle_hard_fairness == candidate.oracle_hard_fairness
        ),
        "request_batches": reference.request_batches == candidate.request_batches,
    }


def _operation_row(replay: Any) -> dict[str, Any]:
    audit = replay.scheduler_control
    proxy = audit["operation_proxy_v2"]
    return {
        "total": int(proxy["total"]),
        "per_query": float(proxy["total"] / len(replay.dispatch_order)),
        "base_control_operations": int(proxy["base_control_operations"]),
        "state_copy_items": int(proxy["state_copy_items"]),
        "cohort_order_items": int(proxy["cohort_order_items"]),
        "frontier_comparisons": int(proxy["frontier_comparisons"]),
        "page_probes": int(audit["page_probes"]),
        "utility_evaluations": int(audit["utility_evaluations"]),
        "feasibility_comparisons": int(audit["feasibility_comparisons"]),
        "retained_state_items": int(audit["retained_state_items"]),
    }


def _evaluate_domain(data: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for model in MODELS:
        replays = {
            name: _run_method(data, model=model, policy=policy, config=None)
            for name, policy in METHOD_SPECS.items()
        }
        runs = []
        for seed_index, seed in enumerate(SEEDS):
            reference = replays["reference"][seed_index]
            transparent = replays["transparent"][seed_index]
            incremental = replays["incremental"][seed_index]
            transparent_checks = _semantic_checks(reference, transparent)
            incremental_checks = _semantic_checks(reference, incremental)
            operations = {
                name: _operation_row(replays[name][seed_index])
                for name in ("transparent", "incremental", "delay_d32")
            }
            runs.append(
                {
                    "seed": seed,
                    "transparent_exact": all(transparent_checks.values()),
                    "incremental_exact": all(incremental_checks.values()),
                    "transparent_checks": transparent_checks,
                    "incremental_checks": incremental_checks,
                    "dispatch_order_sha256": {
                        name: _json_digest(replays[name][seed_index].dispatch_order)
                        for name in ("reference", "transparent", "incremental")
                    },
                    "publication_trace_sha256": {
                        name: _json_digest(
                            replays[name][seed_index].quality_publication_trace
                        )
                        for name in ("reference", "transparent", "incremental")
                    },
                    "operations": operations,
                    "transparent_over_incremental_operation_ratio": (
                        operations["transparent"]["total"]
                        / operations["incremental"]["total"]
                    ),
                    "incremental_over_delay_operation_ratio": (
                        operations["incremental"]["total"]
                        / operations["delay_d32"]["total"]
                    ),
                }
            )
        result[model] = {
            "runs": runs,
            "all_transparent_exact": all(row["transparent_exact"] for row in runs),
            "all_incremental_exact": all(row["incremental_exact"] for row in runs),
            "operation_ratio": {
                "mean": float(
                    np.mean(
                        [
                            row["transparent_over_incremental_operation_ratio"]
                            for row in runs
                        ]
                    )
                ),
                "median": float(
                    np.median(
                        [
                            row["transparent_over_incremental_operation_ratio"]
                            for row in runs
                        ]
                    )
                ),
                "min": float(
                    min(
                        row["transparent_over_incremental_operation_ratio"]
                        for row in runs
                    )
                ),
            },
            "operations_per_query": {
                name: float(
                    np.mean([row["operations"][name]["per_query"] for row in runs])
                )
                for name in ("transparent", "incremental", "delay_d32")
            },
            "retained_state_items": {
                name: int(max(row["operations"][name]["retained_state_items"] for row in runs))
                for name in ("transparent", "incremental", "delay_d32")
            },
        }
    return result


def _deterministic_gate(transfer: dict[str, Any]) -> dict[str, Any]:
    cells = [
        {"domain": domain, "arrival_model": model, **rows[model]}
        for domain, rows in transfer.items()
        for model in MODELS
    ]
    run_rows = [run for cell in cells for run in cell["runs"]]
    cell_ratios = [cell["operation_ratio"]["median"] for cell in cells]
    retained_ratios = [
        cell["retained_state_items"]["incremental"]
        / cell["retained_state_items"]["transparent"]
        for cell in cells
    ]
    checks = {
        "transparent_exact_50_of_50": (
            len(run_rows) == 50 and all(row["transparent_exact"] for row in run_rows)
        ),
        "incremental_exact_50_of_50": (
            len(run_rows) == 50 and all(row["incremental_exact"] for row in run_rows)
        ),
        "median_cell_operation_speedup_at_least_2": float(np.median(cell_ratios))
        >= 2.0,
        "additional_retained_state_under_64_mib_at_8_bytes_per_item": max(
            cell["retained_state_items"]["incremental"]
            - cell["retained_state_items"]["transparent"]
            for cell in cells
        )
        * 8
        < 64 * 1024 * 1024,
    }
    return {
        "decision": (
            "DETERMINISTIC CONTROL PLANE GO"
            if all(checks.values())
            else "DETERMINISTIC CONTROL PLANE NO-GO"
        ),
        "checks": checks,
        "cell_operation_speedup": {
            "mean": float(np.mean(cell_ratios)),
            "median": float(np.median(cell_ratios)),
            "min": float(np.min(cell_ratios)),
            "max": float(np.max(cell_ratios)),
        },
        "retained_state_item_ratio": {
            "median": float(np.median(retained_ratios)),
            "max": float(np.max(retained_ratios)),
        },
        "exact_run_count": sum(row["incremental_exact"] for row in run_rows),
        "expected_run_count": 50,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--domain-matrix-root", type=Path, required=True)
    parser.add_argument("--domain-matrix-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix_reference = json.loads(args.domain_matrix_reference.read_text())
    domains, availability = _load_transfer_domains(
        args.data_root, args.domain_matrix_root, matrix_reference
    )
    if tuple(domains) != TRANSFER_DOMAINS:
        raise RuntimeError("all five frozen domains are required")
    transfer = {domain: _evaluate_domain(domains[domain]) for domain in TRANSFER_DOMAINS}
    gate = _deterministic_gate(transfer)
    report = {
        "schema_version": 1,
        "stage": "deterministic_exact_and_operations",
        "contract_commit": "978063c",
        "inputs": {
            "data_root": str(args.data_root.resolve()),
            "domain_matrix_root": str(args.domain_matrix_root.resolve()),
            "domain_matrix_reference": {
                "path": str(args.domain_matrix_reference.resolve()),
                "sha256": _sha256(args.domain_matrix_reference),
            },
            "availability": availability,
        },
        "methods": METHOD_SPECS,
        "operation_proxy_definition": (
            "events+selection+utility+page probes+feasibility/frontier comparisons+"
            "state-copy items+cohort-order items"
        ),
        "deterministic_gate": gate,
        "transfer": transfer,
        "gpu_used": False,
        "ssh_used": False,
        "claim_boundary": (
            "exact Python control-plane implementation result, not a new scheduler"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": gate["decision"],
                "exact": f"{gate['exact_run_count']}/{gate['expected_run_count']}",
                "operation_speedup": gate["cell_operation_speedup"],
                "retained_state_item_ratio": gate["retained_state_item_ratio"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
