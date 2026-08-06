#!/usr/bin/env python3
"""Evaluate a committed RBRC v0 certificate on one untouched domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.rbrc_v0 import (
    DomainSurface,
    belady_misses,
    orderings,
    quality_summary,
    replay_program,
)
from reprforge.rbrc_v0_inputs import (
    canonical_json_sha256,
    load_bm25_colpali_domain,
    load_irpapers_domain,
    load_omni_domain,
    sha256,
)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _program_spec(
    method: str, certificate: dict[str, Any], floors: tuple[int, ...], reference: int
) -> tuple[str, int, bool]:
    selected = certificate["compiler_output"]["selected"]
    if method == "fixed_top50":
        return "reference", reference, False
    if method == "safety_only":
        floor = selected["static_floor"]
        return ("static", int(floor), False) if floor is not None else (
            "reference",
            reference,
            True,
        )
    if method == "residency_only":
        return "guard", min(floors), False
    if method == "complete_rbrc":
        floor = selected["guard_floor"]
        return ("guard", int(floor), False) if floor is not None else (
            "reference",
            reference,
            True,
        )
    raise ValueError(f"unsupported method: {method}")


def evaluate(
    domain: DomainSurface,
    config: dict[str, Any],
    certificate: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    profile = config["representation_stacks"][profile_name]
    depths = tuple(int(value) for value in profile["primitive_plan_depths"])
    floors, reference = depths[:-1], depths[-1]
    domain.validate(depths)
    online = config["online_contract"]
    quality_contract = config["quality_contract"]
    order_contract = config["orders"]["blind_evaluation"]
    all_orders = orderings(
        domain.query_ids,
        random_orders=int(order_contract["random_permutations"]),
        seed=int(order_contract["seed"]),
    )
    methods = ("fixed_top50", "safety_only", "residency_only", "complete_rbrc")
    cache_baselines = ("lru", "lfu", "gdsf")
    method_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    fixed_rows: dict[str, list[dict[str, Any]]] = {
        policy: [] for policy in cache_baselines
    }
    natural_queries: dict[str, list[dict[str, Any]]] = {}
    capacity = max(1, int(round(domain.corpus_pages * online["capacity_fraction"])))
    epsilon_query = float(quality_contract["query_violation_epsilon"])

    for order_id, query_ids in all_orders:
        for cache_policy in cache_baselines:
            replay = replay_program(
                domain,
                query_ids,
                program="reference",
                floor=reference,
                reference_depth=reference,
                cold_budget=int(online["logical_cold_page_budget"]),
                capacity_fraction=float(online["capacity_fraction"]),
                cache_policy=cache_policy,
            )
            fixed_rows[cache_policy].append(
                {
                    "order_id": order_id,
                    "mean_cold_page_misses": replay["mean_cold_page_misses"],
                    "belady_mean_misses_same_trace": belady_misses(
                        replay["request_trace"], capacity
                    )
                    / len(query_ids),
                }
            )
        for method in methods:
            program, floor, global_abstain = _program_spec(
                method, certificate, floors, reference
            )
            replay = replay_program(
                domain,
                query_ids,
                program=program,
                floor=floor,
                reference_depth=reference,
                cold_budget=int(online["logical_cold_page_budget"]),
                capacity_fraction=float(online["capacity_fraction"]),
                cache_policy=str(online["primary_cache_policy"]),
                force_global_abstain=global_abstain,
            )
            row = {
                "order_id": order_id,
                "program": program,
                "floor": floor,
                "global_abstain": global_abstain,
                "mean_cold_page_misses": replay["mean_cold_page_misses"],
                "mean_depth": replay["mean_depth"],
                "abstain_rate": replay["abstain_rate"],
                "belady_mean_misses_same_trace": belady_misses(
                    replay["request_trace"], capacity
                )
                / len(query_ids),
                **quality_summary(replay["losses"], epsilon_query),
            }
            method_rows[method].append(row)
            if order_id == "natural":
                natural_queries[method] = [
                    {
                        "query_id": query_id,
                        "action_depth": action,
                        "abstained": abstained,
                        "ndcg_at_10": measured,
                        "reference_ndcg_at_10": reference_quality,
                        "signed_regret": reference_quality - measured,
                    }
                    for query_id, action, abstained, measured, reference_quality in zip(
                        replay["query_ids"],
                        replay["actions"],
                        replay["abstained"],
                        replay["quality"],
                        replay["reference_quality"],
                        strict=True,
                    )
                ]

    fixed_summary = {
        policy: {
            "cold_page_misses": _summary(
                [row["mean_cold_page_misses"] for row in rows]
            ),
            "belady_cold_page_misses_same_trace": _summary(
                [row["belady_mean_misses_same_trace"] for row in rows]
            ),
            "orders": rows,
        }
        for policy, rows in fixed_rows.items()
    }
    method_summary: dict[str, Any] = {}
    for method, rows in method_rows.items():
        mean_misses = float(np.mean([row["mean_cold_page_misses"] for row in rows]))
        comparisons = {}
        for policy, baseline in fixed_summary.items():
            baseline_mean = baseline["cold_page_misses"]["mean"]
            comparisons[policy] = {
                "relative_cold_miss_reduction": (baseline_mean - mean_misses)
                / baseline_mean,
                "absolute_cold_miss_reduction": baseline_mean - mean_misses,
            }
        method_summary[method] = {
            "program": rows[0]["program"],
            "floor": rows[0]["floor"],
            "global_abstain": rows[0]["global_abstain"],
            "cold_page_misses": _summary(
                [row["mean_cold_page_misses"] for row in rows]
            ),
            "mean_depth": _summary([row["mean_depth"] for row in rows]),
            "mean_signed_regret": _summary(
                [row["mean_signed_regret"] for row in rows]
            ),
            "quality_violation_rate": _summary(
                [row["quality_violation_rate"] for row in rows]
            ),
            "worst_query_loss": _summary(
                [row["worst_query_loss"] for row in rows]
            ),
            "worst_5pct_cvar": _summary(
                [row["worst_5pct_cvar"] for row in rows]
            ),
            "abstain_rate": _summary([row["abstain_rate"] for row in rows]),
            "gap_to_belady_same_trace": _summary(
                [
                    row["mean_cold_page_misses"]
                    - row["belady_mean_misses_same_trace"]
                    for row in rows
                ]
            ),
            "comparisons_to_fixed_top50": comparisons,
            "orders_with_fewer_misses_than_fixed_top50_lru_fraction": float(
                np.mean(
                    [
                        row["mean_cold_page_misses"]
                        < fixed_rows["lru"][index]["mean_cold_page_misses"]
                        for index, row in enumerate(rows)
                    ]
                )
            ),
            "orders": rows,
        }
    complete = method_summary["complete_rbrc"]
    epsilon_mean = float(quality_contract["mean_signed_regret_epsilon"])
    delta = float(quality_contract["allowed_empirical_violation_rate_delta"])
    gate_checks = {
        "mean_regret_within_epsilon": complete["mean_signed_regret"]["mean"]
        <= epsilon_mean,
        "mean_violation_rate_within_delta": complete["quality_violation_rate"][
            "mean"
        ]
        <= delta,
        "cold_miss_reduction_vs_same_lru_at_least_10pct": complete[
            "comparisons_to_fixed_top50"
        ]["lru"]["relative_cold_miss_reduction"]
        >= 0.10,
        "beats_at_least_one_strong_fixed_cache": any(
            complete["comparisons_to_fixed_top50"][policy][
                "relative_cold_miss_reduction"
            ]
            > 0.0
            for policy in ("lfu", "gdsf")
        ),
        "abstain_rate_at_most_50pct": complete["abstain_rate"]["mean"] <= 0.50,
        "at_least_90pct_orders_reduce_misses": complete[
            "orders_with_fewer_misses_than_fixed_top50_lru_fraction"
        ]
        >= 0.90,
    }
    return {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "profile": profile_name,
        "domain": domain.name,
        "queries": len(domain.query_ids),
        "corpus_pages": domain.corpus_pages,
        "input_sha256": dict(domain.input_sha256),
        "cost_scope": "logical cold high-fidelity pages; no physical latency claim",
        "orders": {
            "count": len(all_orders),
            "natural_plus_random_permutations": True,
            "seed": order_contract["seed"],
        },
        "fixed_top50_cache_baselines": fixed_summary,
        "methods": method_summary,
        "natural_order_per_query": natural_queries,
        "blind_gate": {
            "checks": gate_checks,
            "passes": all(gate_checks.values()),
            "gpu_physical_measurement_permitted": all(gate_checks.values()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--profile", choices=("bm25_colpali", "hpool_full"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--irpapers", nargs=2, metavar=("SURFACE_NPZ", "QUERIES_CSV"))
    source.add_argument("--surface-domain", nargs=2, metavar=("NAME", "ROOT"))
    source.add_argument(
        "--omni-domain", nargs=3, metavar=("NAME", "FAILURE_JSON", "RANKING_TSV")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    config_sha = canonical_json_sha256(args.config)
    if certificate["protocol_id"] != config["protocol_id"]:
        raise ValueError("certificate protocol ID does not match config")
    if certificate["protocol_config_sha256"] != config_sha:
        raise ValueError("certificate was not compiled from this exact frozen config")
    if certificate["profile"] != args.profile:
        raise ValueError("certificate profile mismatch")
    depths = tuple(
        int(value)
        for value in config["representation_stacks"][args.profile][
            "primitive_plan_depths"
        ]
    )
    if args.irpapers:
        if args.profile != "bm25_colpali":
            parser.error("--irpapers is only valid for bm25_colpali")
        surface_path, query_csv = map(Path, args.irpapers)
        locked = config["representation_stacks"][args.profile][
            "method_heldout_domain"
        ]
        if sha256(surface_path) != locked["score_surface_sha256"]:
            raise ValueError("IRPAPERS surface hash differs from frozen protocol")
        if sha256(query_csv) != locked["queries_sha256"]:
            raise ValueError("IRPAPERS query hash differs from frozen protocol")
        domain = load_irpapers_domain("irpapers", surface_path, query_csv, depths)
    elif args.surface_domain:
        name, root = args.surface_domain
        domain = load_bm25_colpali_domain(name, Path(root), depths)
    else:
        assert args.omni_domain is not None
        name, failure, ranking = args.omni_domain
        domain = load_omni_domain(name, Path(failure), Path(ranking), depths)
    result = evaluate(domain, config, certificate, args.profile)
    result["protocol_config_sha256"] = config_sha
    result["certificate_sha256"] = sha256(args.certificate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    complete = result["methods"]["complete_rbrc"]
    print(json.dumps({
        "output": str(args.output),
        "domain": domain.name,
        "complete_rbrc": {
            "program": complete["program"],
            "floor": complete["floor"],
            "mean_cold_page_misses": complete["cold_page_misses"]["mean"],
            "relative_cold_miss_reduction_vs_lru": complete[
                "comparisons_to_fixed_top50"
            ]["lru"]["relative_cold_miss_reduction"],
            "mean_signed_regret": complete["mean_signed_regret"]["mean"],
            "mean_quality_violation_rate": complete["quality_violation_rate"][
                "mean"
            ],
            "mean_abstain_rate": complete["abstain_rate"]["mean"],
        },
        "blind_gate": result["blind_gate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
