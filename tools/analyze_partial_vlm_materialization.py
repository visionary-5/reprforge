#!/usr/bin/env python3
"""Run the frozen partial-VLM-index materialization headroom audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.partial_vlm_inputs import load_irpapers_surface, load_vidore_surface
from reprforge.partial_vlm_materialization import (
    ScoreSurface,
    aggregate_rows,
    evaluate_selection,
    evaluate_text_only,
    evaluate_visual_only,
    fold_assignments,
    gain_recovery,
    online_trace_audit,
    select_pages,
)


POLICIES = (
    "random",
    "corpus_uniform",
    "text_risk",
    "history_frequency",
    "cover25_frequency75",
    "score_oracle",
    "label_rank_oracle",
)
FUSIONS = ("rrf", "zscore")


def _canonical_sha(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_vidore(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=ROOT")
    name, root = value.split("=", 1)
    return name, Path(root)


def _baseline(surface: ScoreSurface, config: dict[str, Any]) -> dict[str, Any]:
    queries = np.arange(surface.queries)
    rrf = config["primary_fusion"]
    all_pages = np.arange(surface.pages)
    return {
        "text_only": evaluate_text_only(surface, queries),
        "visual_only": evaluate_visual_only(surface, queries),
        "full_hybrid": {
            "rrf": evaluate_selection(
                surface,
                queries,
                all_pages,
                fusion="rrf",
                text_top_k=int(rrf["text_top_k"]),
                visual_top_k=int(rrf["visual_top_k"]),
                rrf_constant=int(rrf["rrf_constant"]),
            ),
            "zscore": evaluate_selection(
                surface, queries, all_pages, fusion="zscore"
            ),
        },
    }


def _audit_domain(surface: ScoreSurface, config: dict[str, Any]) -> dict[str, Any]:
    cross_fit = config["cross_fit"]
    folds = int(cross_fit["folds"])
    split = fold_assignments(surface, folds, int(cross_fit["seed"]))
    rrf = config["primary_fusion"]
    baselines = _baseline(surface, config)
    curves: dict[str, Any] = {fusion: {} for fusion in FUSIONS}
    for fusion in FUSIONS:
        for policy in POLICIES:
            policy_rows: dict[str, Any] = {}
            for budget_fraction in config["budgets"]:
                count = int(math.ceil(float(budget_fraction) * surface.pages))
                rows: list[dict[str, Any]] = []
                repetitions = (
                    int(config["random_repetitions"]) if policy == "random" else 1
                )
                for fold in range(folds):
                    history = np.flatnonzero(split != fold)
                    future = np.flatnonzero(split == fold)
                    for repetition in range(repetitions):
                        selected = select_pages(
                            surface,
                            policy=policy,
                            count=count,
                            history_queries=history,
                            future_queries=future,
                            seed=(
                                int(cross_fit["seed"])
                                + 100_000 * fold
                                + 1_000 * repetition
                                + int(round(10_000 * float(budget_fraction)))
                            ),
                        )
                        row = evaluate_selection(
                            surface,
                            future,
                            selected,
                            fusion=fusion,
                            text_top_k=int(rrf["text_top_k"]),
                            visual_top_k=int(rrf["visual_top_k"]),
                            rrf_constant=int(rrf["rrf_constant"]),
                        )
                        rows.append(
                            {
                                key: value
                                for key, value in row.items()
                                if not key.startswith("per_query_")
                            }
                        )
                aggregate = aggregate_rows(rows)
                text_ndcg = baselines["text_only"]["mean_ndcg_at_10"]
                text_recall = baselines["text_only"]["mean_recall_at_100"]
                full = baselines["full_hybrid"][fusion]
                aggregate["ndcg_gain_recovery"] = gain_recovery(
                    aggregate["mean_ndcg_at_10"]["query_weighted_mean"],
                    text_ndcg,
                    full["mean_ndcg_at_10"],
                )
                aggregate["recall_gain_recovery"] = gain_recovery(
                    aggregate["mean_recall_at_100"]["query_weighted_mean"],
                    text_recall,
                    full["mean_recall_at_100"],
                )
                policy_rows[str(budget_fraction)] = aggregate
            curves[fusion][policy] = policy_rows

    online_contract = config["online_trace"]
    online_rows = []
    for order_index in range(int(online_contract["orders"])):
        order = np.arange(surface.queries)
        if order_index > 0:
            np.random.default_rng(
                int(online_contract["seed"]) + order_index - 1
            ).shuffle(order)
        row = online_trace_audit(
            surface,
            order,
            scope_top_k=int(online_contract["query_scope_top_k"]),
            text_top_k=int(rrf["text_top_k"]),
            visual_top_k=int(rrf["visual_top_k"]),
            rrf_constant=int(rrf["rrf_constant"]),
        )
        row["order_id"] = "natural" if order_index == 0 else f"shuffle_{order_index-1:02d}"
        online_rows.append(row)
    online_summary = {
        "orders": online_rows,
        "nonpersistent_mean_ndcg_at_10": float(
            np.mean([row["nonpersistent"]["mean_ndcg_at_10"] for row in online_rows])
        ),
        "persistent_mean_ndcg_at_10": float(
            np.mean([row["persistent"]["mean_ndcg_at_10"] for row in online_rows])
        ),
        "persistent_final_materialized_fraction": float(
            np.mean(
                [
                    row["persistent"]["final_materialized_fraction"]
                    for row in online_rows
                ]
            )
        ),
        "page_event_reuse_fraction": float(
            np.mean(
                [row["amortization"]["page_event_reuse_fraction"] for row in online_rows]
            )
        ),
        "nonpersistent_over_persistent_construction_events": float(
            np.mean(
                [
                    row["amortization"][
                        "nonpersistent_over_persistent_construction_events"
                    ]
                    for row in online_rows
                ]
            )
        ),
    }
    return {
        "queries": surface.queries,
        "corpus_pages": surface.pages,
        "input_sha256": surface.input_sha256,
        "baselines": baselines,
        "cross_fit_fold_sizes": {
            str(fold): int(np.sum(split == fold)) for fold in range(folds)
        },
        "static_materialization_curves": curves,
        "online_promotion_proxy": online_summary,
    }


def _gate(domains: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    required = int(config["audit_gate"]["domains_required_with_positive_headroom"])
    domain_rows = {}
    positive = 0
    for name, domain in domains.items():
        baseline = domain["baselines"]
        text = baseline["text_only"]["mean_ndcg_at_10"]
        full = baseline["full_hybrid"]["rrf"]["mean_ndcg_at_10"]
        full_gain = full - text
        oracle_recoveries = []
        for budget in ("0.2", "0.4"):
            value = domain["static_materialization_curves"]["rrf"][
                "label_rank_oracle"
            ][budget]["ndcg_gain_recovery"]["gain_recovery"]
            if value is not None:
                oracle_recoveries.append(value)
        oracle_best = max(oracle_recoveries) if oracle_recoveries else None
        reuse = domain["online_promotion_proxy"]["page_event_reuse_fraction"]
        checks = {
            "full_hybrid_gain_at_least_0_005": full_gain >= float(
                config["audit_gate"]["minimum_full_hybrid_absolute_gain"]
            ),
            "oracle_gain_recovery_at_least_target": (
                oracle_best is not None
                and oracle_best
                >= float(config["audit_gate"]["oracle_gain_recovery_target"])
            ),
            "persistent_page_reuse_positive": reuse > 0.20,
        }
        if all(checks.values()):
            positive += 1
        domain_rows[name] = {
            "text_ndcg_at_10": text,
            "full_hybrid_ndcg_at_10": full,
            "full_hybrid_minus_text": full_gain,
            "best_label_oracle_gain_recovery_at_20_or_40pct": oracle_best,
            "persistent_page_event_reuse_fraction": reuse,
            "checks": checks,
            "passes_headroom": all(checks.values()),
        }
    return {
        "domains": domain_rows,
        "positive_headroom_domains": positive,
        "required_positive_headroom_domains": required,
        "passes_initial_headroom_gate": positive >= required,
        "warning": "Realizable-policy and faithful DVI physical gates require follow-up; this gate only decides whether the score-surface direction has basic headroom.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vidore", action="append", type=_parse_vidore, default=[])
    parser.add_argument(
        "--irpapers",
        nargs=4,
        metavar=("NAME", "SCORES", "QUERIES", "RUN_MANIFEST"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.vidore and not args.irpapers:
        parser.error("at least one domain is required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    surfaces = [load_vidore_surface(name, root) for name, root in args.vidore]
    if args.irpapers:
        name, scores, queries, manifest = args.irpapers
        surfaces.append(
            load_irpapers_surface(
                name, Path(scores), Path(queries), Path(manifest)
            )
        )
    domains = {surface.name: _audit_domain(surface, config) for surface in surfaces}
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": _canonical_sha(args.config),
        "status": "complete",
        "cost_semantics": "Visual build time/bytes are charged only for selected pages; complete surfaces are an oracle simulator and not a deployable implementation.",
        "domains": domains,
        "initial_headroom_gate": _gate(domains, config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["initial_headroom_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

