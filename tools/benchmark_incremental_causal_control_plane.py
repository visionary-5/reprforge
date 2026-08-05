#!/usr/bin/env python3
"""Empirical single-process timing and memory for causal control planes."""

from __future__ import annotations

import argparse
import gc
import json
import platform
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any

import numpy as np

from reprforge.cagr_faithful_replay import replay_cagr_comparison
from tools.analyze_cagr_bounded_wait import MODELS, _arrival
from tools.analyze_causal_hard_frontier import (
    TRANSFER_DOMAINS,
    _load_transfer_domains,
)
from tools.analyze_multiobjective_oracle_headroom import _sha256


SEED = 20260804
WARMUPS = 2
TIMING_REPEATS = 9
MEMORY_REPEATS = 5
METHODS = {
    "reference": "hard_budget_frontier",
    "transparent": "hard_budget_frontier_transparent",
    "incremental": "hard_budget_frontier_incremental",
    "delay_d32": "delay_scheduling_d32_control",
}


def _trace(
    data: dict[str, Any], model: str, size: str
) -> tuple[list[list[int]], np.ndarray, np.ndarray, np.ndarray]:
    order, times = _arrival(data, SEED, model)
    gains = np.asarray(data["quality_gain"], dtype=np.float64)
    if size == "large":
        return data["cohorts"], order, times, gains
    if size != "small":
        raise ValueError(size)
    query_ids = order[:64]
    return (
        [data["cohorts"][int(query)] for query in query_ids],
        np.arange(len(query_ids), dtype=np.int64),
        times[:64] - times[0],
        gains[query_ids],
    )


def _run(
    data: dict[str, Any],
    model: str,
    size: str,
    policy: str,
) -> Any:
    cohorts, order, times, gains = _trace(data, model, size)
    return replay_cagr_comparison(
        cohorts,
        order,
        times,
        gains,
        base_mean_quality=float(data["base_mean_quality"]),
        corpus_pages=int(data["corpus_pages"]),
        request_batch_size=8,
        window=64,
        policy=policy,
        cache_capacity=80,
        arrival_clock="unit",
    )


def _summary(values: list[int]) -> dict[str, float | list[int]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "raw": values,
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _time_workload(
    data: dict[str, Any], model: str, size: str
) -> dict[str, Any]:
    method_names = tuple(METHODS)
    for repeat in range(WARMUPS):
        order = method_names[repeat:] + method_names[:repeat]
        for name in order:
            _run(data, model, size, METHODS[name])
    timings = {name: [] for name in method_names}
    for repeat in range(TIMING_REPEATS):
        order = method_names[repeat % len(method_names) :] + method_names[
            : repeat % len(method_names)
        ]
        for name in order:
            gc.collect()
            gc.disable()
            start = time.perf_counter_ns()
            try:
                _run(data, model, size, METHODS[name])
            finally:
                elapsed = time.perf_counter_ns() - start
                gc.enable()
            timings[name].append(elapsed)
    summaries = {name: _summary(values) for name, values in timings.items()}
    return {
        "warmups": WARMUPS,
        "repeats": TIMING_REPEATS,
        "methods": summaries,
        "transparent_over_incremental_median_speedup": (
            summaries["transparent"]["median"]
            / summaries["incremental"]["median"]
        ),
        "reference_over_incremental_median_speedup": (
            summaries["reference"]["median"]
            / summaries["incremental"]["median"]
        ),
        "incremental_over_delay_median_ratio": (
            summaries["incremental"]["median"]
            / summaries["delay_d32"]["median"]
        ),
    }


def _memory_workload(
    data: dict[str, Any], model: str, size: str
) -> dict[str, Any]:
    peaks = {name: [] for name in METHODS}
    for repeat in range(MEMORY_REPEATS):
        names = tuple(METHODS)
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        for name in order:
            gc.collect()
            tracemalloc.start()
            try:
                _run(data, model, size, METHODS[name])
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            peaks[name].append(int(peak))
    summaries = {name: _summary(values) for name, values in peaks.items()}
    return {
        "repeats": MEMORY_REPEATS,
        "methods": summaries,
        "incremental_over_transparent_median_peak_ratio": (
            summaries["incremental"]["median"]
            / summaries["transparent"]["median"]
        ),
        "incremental_over_reference_median_peak_ratio": (
            summaries["incremental"]["median"]
            / summaries["reference"]["median"]
        ),
    }


def _empirical_gate(
    workloads: dict[str, Any], deterministic: dict[str, Any]
) -> dict[str, Any]:
    large_speedups = [
        row["timing"]["transparent_over_incremental_median_speedup"]
        for row in workloads.values()
        if row["size"] == "large"
    ]
    reference_large_speedups = [
        row["timing"]["reference_over_incremental_median_speedup"]
        for row in workloads.values()
        if row["size"] == "large"
    ]
    small_slowdown = [
        1.0 / row["timing"]["transparent_over_incremental_median_speedup"]
        for row in workloads.values()
        if row["size"] == "small"
    ]
    reference_small_slowdown = [
        1.0 / row["timing"]["reference_over_incremental_median_speedup"]
        for row in workloads.values()
        if row["size"] == "small"
    ]
    memory_ratios = [
        row["memory"]["incremental_over_transparent_median_peak_ratio"]
        for row in workloads.values()
    ]
    reference_memory_ratios = [
        row["memory"]["incremental_over_reference_median_peak_ratio"]
        for row in workloads.values()
    ]
    deterministic_gate = deterministic["deterministic_gate"]
    checks = {
        "exact_50_of_50": deterministic_gate["checks"][
            "incremental_exact_50_of_50"
        ],
        "operation_or_cpu_main_gate": bool(
            deterministic_gate["checks"][
                "median_cell_operation_speedup_at_least_2"
            ]
            or np.median(large_speedups) >= 1.5
        ),
        "no_small_trace_slowdown_over_10_percent": max(small_slowdown) <= 1.10,
        "median_tracemalloc_peak_ratio_at_most_1_25": float(
            np.median(memory_ratios)
        )
        <= 1.25,
        "additional_retained_state_under_64_mib": deterministic_gate["checks"][
            "additional_retained_state_under_64_mib_at_8_bytes_per_item"
        ],
    }
    return {
        "decision": (
            "INCREMENTAL CONTROL PLANE GO"
            if all(checks.values())
            else "INCREMENTAL CONTROL PLANE NO-GO"
        ),
        "checks": checks,
        "registered_transparent_over_incremental_large_cpu_speedup": {
            "median": float(np.median(large_speedups)),
            "p95": float(np.quantile(large_speedups, 0.95)),
            "min": float(np.min(large_speedups)),
            "max": float(np.max(large_speedups)),
        },
        "registered_incremental_over_transparent_small_cpu_ratio": {
            "median": float(np.median(small_slowdown)),
            "max": float(np.max(small_slowdown)),
        },
        "registered_tracemalloc_incremental_over_transparent": {
            "median": float(np.median(memory_ratios)),
            "p95": float(np.quantile(memory_ratios, 0.95)),
            "max": float(np.max(memory_ratios)),
        },
        "uninstrumented_reference_sensitivity_audit": {
            "reference_over_incremental_large_cpu_speedup": {
                "median": float(np.median(reference_large_speedups)),
                "min": float(np.min(reference_large_speedups)),
                "max": float(np.max(reference_large_speedups)),
            },
            "incremental_over_reference_small_cpu_ratio": {
                "median": float(np.median(reference_small_slowdown)),
                "max": float(np.max(reference_small_slowdown)),
            },
            "tracemalloc_incremental_over_reference": {
                "median": float(np.median(reference_memory_ratios)),
                "max": float(np.max(reference_memory_ratios)),
            },
            "used_for_registered_gate": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--domain-matrix-root", type=Path, required=True)
    parser.add_argument("--domain-matrix-reference", type=Path, required=True)
    parser.add_argument("--deterministic-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix_reference = json.loads(args.domain_matrix_reference.read_text())
    deterministic = json.loads(args.deterministic_result.read_text())
    domains, availability = _load_transfer_domains(
        args.data_root, args.domain_matrix_root, matrix_reference
    )
    if tuple(domains) != TRANSFER_DOMAINS:
        raise RuntimeError("all five frozen domains are required")
    workloads = {}
    for domain in TRANSFER_DOMAINS:
        for model in MODELS:
            for size in ("small", "large"):
                key = f"{domain}/{model}/{size}"
                data = domains[domain]
                trace = _trace(data, model, size)
                workloads[key] = {
                    "domain": domain,
                    "arrival_model": model,
                    "size": size,
                    "query_count": len(trace[0]),
                    "corpus_pages": int(data["corpus_pages"]),
                    "candidate_union_pages": len(
                        set().union(*(set(cohort) for cohort in trace[0]))
                    ),
                    "timing": _time_workload(data, model, size),
                    "memory": _memory_workload(data, model, size),
                }
                print(f"completed {key}", flush=True)
    gate = _empirical_gate(workloads, deterministic)
    report = {
        "schema_version": 1,
        "stage": "empirical_cpu_and_memory",
        "contract_commit": "978063c",
        "protocol": {
            "timer": "time.perf_counter_ns",
            "single_python_process": True,
            "gc_disabled_during_timed_replay": True,
            "seed": SEED,
            "warmups": WARMUPS,
            "timing_repeats": TIMING_REPEATS,
            "memory_repeats": MEMORY_REPEATS,
            "method_order": "fixed rotation",
            "small_query_count": 64,
            "large_query_count": "full domain trace",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "inputs": {
            "deterministic_result": {
                "path": str(args.deterministic_result.resolve()),
                "sha256": _sha256(args.deterministic_result),
            },
            "availability": availability,
        },
        "empirical_gate": gate,
        "workloads": workloads,
        "claim_boundary": (
            "local single-process Python wall time; nondeterministic and not production latency"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
