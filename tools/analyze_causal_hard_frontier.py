#!/usr/bin/env python3
"""Materialize B32 headroom as a causal policy and run frozen transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from reprforge.candidate_fusion import _candidate_ndcg
from reprforge.intervention_utility import _ndcg_row
from reprforge.progressive_oracle import rank_order
from tools.analyze_cagr_bounded_wait import MODELS, SEEDS, _json_digest
from tools.analyze_hard_fair_oracle import (
    BASE_ORACLE_CONFIG,
    _hard_fair_aggregate,
)
from tools.analyze_multiobjective_oracle_headroom import (
    _endpoint_specs,
    _horizons,
    _run_method,
    _sha256,
    _summarize_method,
)
from tools.analyze_windowed_arrivals import load_domain


REFERENCE_CONFIG = {**BASE_ORACLE_CONFIG, "bypass_budget": 32}
METHOD_SPECS = {
    "fifo": {"policy": "fifo"},
    "frontier": {"policy": "frontier"},
    "overlap_only": {"policy": "overlap_only"},
    "bounded_cagr": _endpoint_specs()["bounded_cagr"],
    "hard_budget_frontier": {"policy": "hard_budget_frontier"},
}
TRANSFER_DOMAINS = (
    "hr",
    "finance",
    "computer_science",
    "industrial",
    "pharmaceuticals",
)


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _control_aggregate(replays: list[Any]) -> dict[str, Any]:
    audits = [replay.scheduler_control for replay in replays]
    query_count = sum(len(replay.dispatch_order) for replay in replays)
    additive = (
        "arrival_events",
        "timer_events",
        "dispatch_events",
        "selection_count",
        "forced_selection_count",
        "protected_unique_queries",
        "utility_evaluations",
        "page_probes",
        "feasibility_comparisons",
        "control_operations",
    )
    totals = {
        key: sum(int(audit.get(key, 0)) for audit in audits) for key in additive
    }
    return {
        "runs": len(replays),
        "policy_names": sorted({str(audit["policy"]) for audit in audits}),
        "counting_scopes": sorted(
            {str(audit["counting_scope"]) for audit in audits}
        ),
        **totals,
        "control_operations_per_query": (
            totals["control_operations"] / query_count if query_count else 0.0
        ),
        "fixed_configs": sorted(
            {
                json.dumps(audit["fixed_config"], sort_keys=True)
                for audit in audits
                if "fixed_config" in audit
            }
        ),
    }


def _extended_summary(
    data: dict[str, Any],
    replays: list[Any],
    horizons_by_seed: list[dict[str, float]],
) -> dict[str, Any]:
    result = _summarize_method(data, replays, horizons_by_seed)
    sojourn = np.concatenate(
        [np.asarray(replay.sojourn_unit_time, dtype=np.float64) for replay in replays]
    )
    wait = np.concatenate(
        [np.asarray(replay.wait_unit_time, dtype=np.float64) for replay in replays]
    )
    bypass = np.concatenate(
        [np.asarray(replay.bypass_count, dtype=np.int64) for replay in replays]
    )
    result["tails"] = {
        "sojourn_unit_time": _distribution(sojourn),
        "wait_unit_time": _distribution(wait),
        "younger_bypass": _distribution(bypass.astype(np.float64)),
        "bypass_at_least_64_fraction": float(np.mean(bypass >= 64)),
    }
    result["hard_fairness"] = _hard_fair_aggregate(replays)
    result["control_plane"] = _control_aggregate(replays)
    return result


def _load_flat_domain(
    root: Path, expected: dict[str, Any], candidate_k: int = 20
) -> dict[str, Any]:
    paths = {
        "bm25": root / "bm25-runtime.npz",
        "visual": root / "visual-runtime.npz",
        "labels": root / "oracle-labels.npz",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256(path)
        expected_digest = expected[name]["sha256"]
        if observed != expected_digest:
            raise ValueError(
                f"{name} digest mismatch: observed={observed} expected={expected_digest}"
            )
    with np.load(paths["bm25"], allow_pickle=False) as payload:
        text_values = {name: payload[name] for name in payload.files}
    with np.load(paths["visual"], allow_pickle=False) as payload:
        visual_values = {name: payload[name] for name in payload.files}
    with np.load(paths["labels"], allow_pickle=False) as payload:
        labels = {name: payload[name] for name in payload.files}
    if not np.array_equal(text_values["query_ids"], visual_values["query_ids"]):
        raise ValueError(f"query identifiers differ in {root}")
    if not np.array_equal(text_values["corpus_ids"], visual_values["corpus_ids"]):
        raise ValueError(f"corpus identifiers differ in {root}")
    text_scores = np.asarray(text_values["scores"], dtype=np.float64)
    visual_scores = np.asarray(visual_values["scores"], dtype=np.float64)
    corpus_ids = np.asarray(text_values["corpus_ids"])
    qrels = np.zeros(text_scores.shape, dtype=np.int16)
    qrels[
        np.asarray(labels["query_positions"], dtype=np.int64),
        np.asarray(labels["corpus_positions"], dtype=np.int64),
    ] = np.asarray(labels["relevance"], dtype=np.int16)
    if np.any(qrels.max(axis=1) == 0):
        raise ValueError(f"at least one query has no qrel in {root}")
    text_order = rank_order(text_scores, corpus_ids)
    candidate_k = min(candidate_k, text_scores.shape[1])
    base_quality = np.asarray(
        [
            _ndcg_row(text_scores[q], qrels[q], corpus_ids, cutoff=10)
            for q in range(text_scores.shape[0])
        ],
        dtype=np.float64,
    )
    text = SimpleNamespace(scores=text_scores, corpus_ids=corpus_ids)
    visual = SimpleNamespace(scores=visual_scores, corpus_ids=corpus_ids)
    refined_quality = _candidate_ndcg(
        text,
        visual,
        qrels,
        text_order,
        candidate_k=candidate_k,
        method="zscore_sum",
        cutoff=10,
    )
    return {
        "cohorts": text_order[:, :candidate_k].tolist(),
        "quality_gain": refined_quality - base_quality,
        "base_mean_quality": float(base_quality.mean()),
        "refined_mean_quality": float(refined_quality.mean()),
        "query_count": int(text_scores.shape[0]),
        "corpus_pages": int(text_scores.shape[1]),
        "candidate_union_pages": int(
            len(set(text_order[:, :candidate_k].reshape(-1).tolist()))
        ),
        "trace_layout": "frozen-domain-matrix-flat",
        "provenance": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "use": (
                    "post-hoc quality accounting only"
                    if name in {"visual", "labels"}
                    else "locator cohort construction"
                ),
            }
            for name, path in paths.items()
        },
    }


def _verify_manifest_domain(
    data: dict[str, Any], expected: dict[str, Any]
) -> None:
    observed = data["provenance"]
    mapping = {
        "bm25": "text_runtime",
        "visual": "visual_runtime",
        "labels": "oracle_labels",
    }
    for expected_name, observed_name in mapping.items():
        if observed[observed_name]["sha256"] != expected[expected_name]["sha256"]:
            raise ValueError(f"frozen digest mismatch for {expected_name}")


def _load_transfer_domains(
    data_root: Path,
    matrix_root: Path,
    matrix_reference: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    loaded: dict[str, dict[str, Any]] = {}
    availability: dict[str, dict[str, Any]] = {}
    reference_domains = matrix_reference["domains"]
    for domain in TRANSFER_DOMAINS:
        try:
            if domain in {"hr", "finance"}:
                data = load_domain(data_root / domain, 20)
                _verify_manifest_domain(data, reference_domains[domain]["artifacts"])
            else:
                root_name = domain.replace("_", "-")
                data = _load_flat_domain(
                    matrix_root / root_name,
                    reference_domains[domain]["artifacts"],
                )
            loaded[domain] = data
            availability[domain] = {
                "status": "available",
                "query_count": data["query_count"],
                "corpus_pages": data["corpus_pages"],
                "candidate_union_pages": data["candidate_union_pages"],
                "provenance": data["provenance"],
            }
        except (FileNotFoundError, KeyError, ValueError) as error:
            availability[domain] = {
                "status": "unavailable_or_digest_mismatch",
                "error": str(error),
            }
    return loaded, availability


def _equivalence_cell(reference: Any, causal: Any, seed: int) -> dict[str, Any]:
    checks = {
        "dispatch_order": reference.dispatch_order == causal.dispatch_order,
        "completion_pages": reference.completion_pages == causal.completion_pages,
        "completion_elapsed": (
            reference.completion_unit_cost == causal.completion_unit_cost
        ),
        "sojourn_elapsed": reference.sojourn_unit_time == causal.sojourn_unit_time,
        "charged_work": reference.total_unit_work == causal.total_unit_work,
        "cache": reference.cache == causal.cache,
        "final_union": reference.final_unique_pages == causal.final_unique_pages,
        "bypass": reference.bypass_count == causal.bypass_count,
        "publication_trace": (
            reference.quality_publication_trace == causal.quality_publication_trace
        ),
    }
    return {
        "seed": seed,
        "checks": checks,
        "byte_equivalent": all(checks.values()),
        "reference_order_sha256": _json_digest(reference.dispatch_order),
        "causal_order_sha256": _json_digest(causal.dispatch_order),
        "reference_trace_sha256": _json_digest(
            reference.quality_publication_trace
        ),
        "causal_trace_sha256": _json_digest(causal.quality_publication_trace),
    }


def _evaluate_equivalence(domains: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evaluation = {}
    cells = []
    for domain in ("hr", "finance"):
        evaluation[domain] = {}
        data = domains[domain]
        for model in MODELS:
            references = _run_method(
                data,
                model=model,
                policy="multiobjective_oracle",
                config=REFERENCE_CONFIG,
            )
            causal = _run_method(
                data,
                model=model,
                policy="hard_budget_frontier",
                config=None,
            )
            rows = [
                _equivalence_cell(reference, method, seed)
                for seed, reference, method in zip(SEEDS, references, causal)
            ]
            cells.extend(rows)
            evaluation[domain][model] = rows
    return {
        "expected_cell_count": 20,
        "observed_cell_count": len(cells),
        "equivalent_cell_count": sum(row["byte_equivalent"] for row in cells),
        "all_cells_byte_equivalent": (
            len(cells) == 20 and all(row["byte_equivalent"] for row in cells)
        ),
        "evaluation": evaluation,
    }


def _evaluate_transfer_domain(data: dict[str, Any]) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        replay_sets = {
            name: _run_method(
                data,
                model=model,
                policy=spec["policy"],
                config=spec.get("config"),
            )
            for name, spec in METHOD_SPECS.items()
        }
        horizons_by_seed = [
            _horizons(replay_sets, seed_index) for seed_index in range(len(SEEDS))
        ]
        by_model[model] = {
            "common_horizons_by_seed": [
                {"seed": seed, **horizon}
                for seed, horizon in zip(SEEDS, horizons_by_seed)
            ],
            "methods": {
                name: _extended_summary(data, replays, horizons_by_seed)
                for name, replays in replay_sets.items()
            },
        }
    return by_model


def _modern_availability(audit_path: Path) -> dict[str, Any]:
    if not audit_path.is_file():
        return {
            "status": "not_run_missing_local_replay_trace",
            "audit_path": str(audit_path.resolve()),
            "reason": "artifact audit itself is not locally readable",
        }
    audit = json.loads(audit_path.read_text())
    rows = {}
    any_local = False
    for domain, metadata in audit.get("valid_runs", {}).items():
        manifest = Path(metadata["manifest_path"])
        local = manifest.is_file()
        any_local = any_local or local
        rows[domain] = {
            "manifest_path": str(manifest),
            "manifest_sha256_expected": metadata.get("manifest_sha256"),
            "locally_readable": local,
            "runtime_sha256": metadata.get("runtime_sha256"),
            "oracle_labels_sha256": metadata.get("oracle_labels_sha256"),
        }
    return {
        "status": (
            "not_run_missing_supported_loader"
            if any_local
            else "not_run_missing_local_replay_trace"
        ),
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": _sha256(audit_path),
        "known_artifacts": rows,
        "required_components": [
            "locally readable per-query Top-20 cohorts",
            "corpus identity",
            "post-hoc quality gains",
        ],
        "download_or_gpu_used": False,
    }


def _comparison_summary(transfer: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, list[dict[str, float]]] = {
        name: [] for name in METHOD_SPECS
    }
    for domain_result in transfer.values():
        for model in MODELS:
            methods = domain_result[model]["methods"]
            bounded = methods["bounded_cagr"]
            frontier = methods["frontier"]
            for name, method in methods.items():
                rows[name].append(
                    {
                        "mean_sojourn_ratio": (
                            method["system"]["sojourn_unit_time"]["mean"]
                            / bounded["system"]["sojourn_unit_time"]["mean"]
                        ),
                        "work_ratio": (
                            method["system"]["unit_work_per_query"]
                            / bounded["system"]["unit_work_per_query"]
                        ),
                        "elapsed_regret_ratio": (
                            method["axes"]["elapsed_unit_time"][
                                "normalized_quality_regret_auc"
                            ]["mean"]
                            / frontier["axes"]["elapsed_unit_time"][
                                "normalized_quality_regret_auc"
                            ]["mean"]
                        ),
                        "unique_page_regret": method["axes"][
                            "unique_compiled_pages"
                        ]["normalized_quality_regret_auc"]["mean"],
                        "p99_sojourn": method["tails"]["sojourn_unit_time"][
                            "p99"
                        ],
                        "control_operations_per_query": method["control_plane"][
                            "control_operations_per_query"
                        ],
                    }
                )
    return {
        name: {
            metric: {
                "mean": float(np.mean([row[metric] for row in method_rows])),
                "median": float(np.median([row[metric] for row in method_rows])),
                "min": float(np.min([row[metric] for row in method_rows])),
                "max": float(np.max([row[metric] for row in method_rows])),
            }
            for metric in method_rows[0]
        }
        for name, method_rows in rows.items()
        if method_rows
    }


def _paper_gate(
    equivalence: dict[str, Any],
    transfer: dict[str, Any],
    availability: dict[str, Any],
    modern: dict[str, Any],
) -> dict[str, Any]:
    cells = []
    parity_and_budget = True
    for domain, domain_result in transfer.items():
        expected_union = availability[domain]["candidate_union_pages"]
        for model in MODELS:
            methods = domain_result[model]["methods"]
            hard = methods["hard_budget_frontier"]
            bounded = methods["bounded_cagr"]
            frontier = methods["frontier"]
            ratios = {
                "mean_sojourn_over_bounded": (
                    hard["system"]["sojourn_unit_time"]["mean"]
                    / bounded["system"]["sojourn_unit_time"]["mean"]
                ),
                "work_over_bounded": (
                    hard["system"]["unit_work_per_query"]
                    / bounded["system"]["unit_work_per_query"]
                ),
                "elapsed_regret_over_frontier": (
                    hard["axes"]["elapsed_unit_time"][
                        "normalized_quality_regret_auc"
                    ]["mean"]
                    / frontier["axes"]["elapsed_unit_time"][
                        "normalized_quality_regret_auc"
                    ]["mean"]
                ),
            }
            p99_ratio = hard["tails"]["sojourn_unit_time"]["p99"] / min(
                bounded["tails"]["sojourn_unit_time"]["p99"],
                frontier["tails"]["sojourn_unit_time"]["p99"],
            )
            parity = bool(
                hard["system"]["dispatch_complete"]
                and hard["system"]["final_union_pages"] == [expected_union]
            )
            violations = hard["hard_fairness"]["budget_violation_count"]
            parity_and_budget = parity_and_budget and parity and violations == 0
            cells.append(
                {
                    "domain": domain,
                    "arrival_model": model,
                    "ratios": ratios,
                    "p99_over_best_endpoint": p99_ratio,
                    "parity": parity,
                    "budget_violation_count": violations,
                    "at_least_one_primary_improves_5_percent": (
                        min(ratios.values()) <= 0.95
                    ),
                }
            )
    available_count = sum(
        row["status"] == "available" for row in availability.values()
    )
    ratio_names = (
        "mean_sojourn_over_bounded",
        "work_over_bounded",
        "elapsed_regret_over_frontier",
    )
    medians = {
        name: float(np.median([cell["ratios"][name] for cell in cells]))
        for name in ratio_names
    }
    no_catastrophe = all(
        value <= 1.10 for cell in cells for value in cell["ratios"].values()
    )
    improvement_fraction = float(
        np.mean([cell["at_least_one_primary_improves_5_percent"] for cell in cells])
    )
    p99_safe_fraction = float(
        np.mean([cell["p99_over_best_endpoint"] <= 1.05 for cell in cells])
    )
    checks = {
        "reference_equivalence_20_of_20": equivalence[
            "all_cells_byte_equivalent"
        ],
        "at_least_four_vidore_domains_available": available_count >= 4,
        "all_run_cells_parity_and_b32_safe": parity_and_budget,
        "no_primary_ratio_exceeds_1.10": no_catastrophe,
        "all_cross_cell_median_primary_ratios_at_most_1": all(
            value <= 1.0 for value in medians.values()
        ),
        "at_least_60_percent_cells_improve_one_axis_5_percent": (
            improvement_fraction >= 0.60
        ),
        "p99_safe_in_at_least_80_percent_cells": p99_safe_fraction >= 0.80,
    }
    passed = all(checks.values())
    modern_verified = modern["status"] == "completed"
    return {
        "decision": (
            (
                "PAPER METHOD CANDIDATE"
                if modern_verified
                else "PAPER METHOD CANDIDATE; CROSS-RETRIEVER UNVERIFIED"
            )
            if passed
            else "NOT YET A PAPER METHOD CANDIDATE"
        ),
        "checks": checks,
        "available_vidore_domain_count": available_count,
        "evaluated_domain_arrival_cells": len(cells),
        "median_primary_ratios": medians,
        "cell_improvement_fraction": improvement_fraction,
        "p99_safe_cell_fraction": p99_safe_fraction,
        "modern_transfer_status": modern["status"],
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--domain-matrix-root", type=Path, required=True)
    parser.add_argument("--domain-matrix-reference", type=Path, required=True)
    parser.add_argument("--modern-artifact-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix_reference = json.loads(args.domain_matrix_reference.read_text())
    domains, availability = _load_transfer_domains(
        args.data_root, args.domain_matrix_root, matrix_reference
    )
    missing_core = [domain for domain in ("hr", "finance") if domain not in domains]
    if missing_core:
        raise RuntimeError(f"missing core equivalence domains: {missing_core}")
    equivalence = _evaluate_equivalence(domains)
    transfer = {
        domain: _evaluate_transfer_domain(data)
        for domain, data in domains.items()
    }
    modern = _modern_availability(args.modern_artifact_audit)
    comparison_summary = _comparison_summary(transfer)
    gate = _paper_gate(equivalence, transfer, availability, modern)
    report = {
        "schema_version": 1,
        "stage": "causal-hard-frontier-materialization-and-frozen-transfer",
        "contract_commit": "f0ab00e",
        "gpu_used": False,
        "downloads_performed": False,
        "policy": {
            "name": "hard_budget_frontier",
            "qrel_or_quality_input": False,
            "future_arrival_array_input": False,
            "future_arrival_content_input": False,
            "reference_config": REFERENCE_CONFIG,
        },
        "inputs": {
            "data_root": str(args.data_root.resolve()),
            "domain_matrix_root": str(args.domain_matrix_root.resolve()),
            "domain_matrix_reference": {
                "path": str(args.domain_matrix_reference.resolve()),
                "sha256": _sha256(args.domain_matrix_reference),
            },
            "modern_artifact_audit": {
                "path": str(args.modern_artifact_audit.resolve()),
                "sha256": _sha256(args.modern_artifact_audit),
            },
        },
        "availability": availability,
        "reference_equivalence": equivalence,
        "transfer": transfer,
        "cross_domain_comparison_summary": comparison_summary,
        "modern_transfer": modern,
        "paper_method_gate": gate,
        "claim_boundary": (
            "causal replay-level method candidate only; ViDoRe arrivals are stress "
            "models, not production chronology; no GPU throughput claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "equivalence": {
                    key: value
                    for key, value in equivalence.items()
                    if key != "evaluation"
                },
                "availability": {
                    key: value["status"] for key, value in availability.items()
                },
                "modern": modern["status"],
                "gate": {
                    key: value for key, value in gate.items() if key != "cells"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
