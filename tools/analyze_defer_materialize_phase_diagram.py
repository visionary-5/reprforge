#!/usr/bin/env python3
"""Build a measured defer--partial--Full operating-region diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reprforge.defer_materialize_phase import (
    break_even_queries,
    history_policy_no_regression,
    smallest_oracle_quality_plan,
    winner_grid,
)


ALIASES = {"pharmaceuticals": "pharma", "industrial": "industrial"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load(args.config)
    quality = config["quality_gate"]
    axes = config["axes"]
    domains = {}
    for domain in config["domains"]:
        oracle_path = args.repo_root / config["input_artifacts"]["oracle"].format(domain=domain)
        verifier_path = args.repo_root / config["input_artifacts"]["verifier"].format(
            domain_alias=ALIASES[domain]
        )
        oracle = load(oracle_path)
        verifier = load(verifier_path)
        full = oracle["baselines"]["full_bm25_colsmol_omni"]
        plan = smallest_oracle_quality_plan(
            oracle["curves"]["greedy_marginal_ndcg_oracle"],
            full_hit=float(full["query_hit_at_20"]),
            minimum_gain_recovery=float(quality["oracle_minimum_full_gain_recovery"]),
            maximum_hit_loss=float(quality["oracle_maximum_query_hit_at_20_loss_vs_full"]),
        )
        if plan is None:
            raise RuntimeError(f"no oracle quality plan for {domain}")
        costs = oracle["measured_cost_reference"]
        page_seconds = float(verifier["verifier"]["page_seconds_mean"])
        partial_build = float(plan["projected_cost"]["incremental_omni_build_seconds"])
        colsmol_build = float(costs["base_colsmol_build_seconds"])
        full_build = float(costs["full_omni_build_seconds"])
        current_build = colsmol_build + partial_build
        break_even = {}
        for avoided in map(int, axes["avoided_page_checks_per_query"]):
            break_even[str(avoided)] = {
                "oracle_partial_vs_dvi_queries": break_even_queries(partial_build, page_seconds, avoided),
                "current_stack_vs_dvi_queries": break_even_queries(current_build, page_seconds, avoided),
                "full_vs_dvi_queries": break_even_queries(full_build, page_seconds, avoided),
            }
        grid = winner_grid(
            horizons=list(map(int, axes["query_horizons"])),
            avoided_pages_values=list(map(int, axes["avoided_page_checks_per_query"])),
            verifier_page_seconds=page_seconds,
            oracle_partial_build_seconds=partial_build,
            current_stack_build_seconds=current_build,
            full_build_seconds=full_build,
        )
        for row in grid:
            costs_row = row["costs_gpu_seconds"]
            deployable_names = ("dvi_defer", "current_colsmol_partial", "full_omni")
            row["current_stack_winner"] = min(
                deployable_names, key=lambda name: (costs_row[name], name)
            )
        history = history_policy_no_regression(oracle["history_residual_crossfit_curves"])
        oracle_wins = any(row["winner"] == "oracle_partial" for row in grid)
        current_wins = any(
            row["current_stack_winner"] == "current_colsmol_partial" for row in grid
        )
        full_wins = any(row["current_stack_winner"] == "full_omni" for row in grid)
        dvi_wins = any(row["current_stack_winner"] == "dvi_defer" for row in grid)
        full_bytes = float(costs["full_omni_index_bytes"])
        current_bytes = float(costs["base_colsmol_index_bytes"]) + float(
            plan["projected_cost"]["incremental_omni_index_bytes"]
        )
        domains[domain] = {
            "input_sha256": {"oracle": sha256(oracle_path), "verifier": sha256(verifier_path)},
            "measured": {
                "verifier_gpu_forward_seconds_per_query_page": page_seconds,
                "full_omni_build_seconds": full_build,
                "full_omni_index_bytes": full_bytes,
                "full_colsmol_build_seconds": colsmol_build,
                "full_colsmol_index_bytes": float(costs["base_colsmol_index_bytes"]),
            },
            "oracle_quality_plan": plan,
            "oracle_partial_build_seconds": partial_build,
            "current_stack_build_seconds": current_build,
            "current_stack_build_seconds_over_full": current_build / full_build,
            "current_stack_index_bytes": current_bytes,
            "current_stack_index_bytes_over_full": current_bytes / full_bytes,
            "history_transfer": history,
            "break_even_queries": break_even,
            "grid": grid,
            "winner_presence": {
                "dvi_defer_current_stack_comparison": dvi_wins,
                "oracle_partial": oracle_wins,
                "current_colsmol_partial": current_wins,
                "full_omni_current_stack_comparison": full_wins,
            },
            "checks": {
                "oracle_middle_region_exists": oracle_wins,
                "realizable_history_policy_no_regression": history["any_nonzero_budget_passes"],
                "current_stack_non_dominated_in_build_time": current_build < full_build,
                "current_stack_non_dominated_in_storage": current_bytes < full_bytes,
            },
        }
    passes = all(
        row["checks"]["oracle_middle_region_exists"]
        and row["checks"]["realizable_history_policy_no_regression"]
        and row["checks"]["current_stack_non_dominated_in_build_time"]
        for row in domains.values()
    )
    output = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "GO" if passes else "NO_GO_CURRENT_METHOD",
        "config_sha256": sha256(args.config),
        "domains": domains,
        "decision": {
            "passes_current_method_gate": passes,
            "interpretation": (
                "A realizable middle operating region is supported by the current stack."
                if passes
                else "Oracle headroom may exist, but the current full-corpus ColSmol plus partial Omni stack does not establish a transferable middle operating region."
            ),
        },
        "warnings": [
            "Full is not guaranteed to become optimal from query volume alone. If a fixed partial working set preserves quality at equal query cost, it remains cheaper than Full at every horizon.",
            "Full is the high-load endpoint only when visual demand broadens toward the corpus, quality requires broader coverage, or online promotion violates a tail-latency objective.",
            "The winner grid gives the oracle the same persistent query cost as Full and is deliberately favorable to partial materialization.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "domains": {d: {"checks": v["checks"], "winner_presence": v["winner_presence"]} for d, v in domains.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
