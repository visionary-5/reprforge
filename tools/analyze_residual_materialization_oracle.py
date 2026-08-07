#!/usr/bin/env python3
"""Run the frozen BM25+ColSmol -> selective Omni materialization oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.partial_page_selector import PageRiskFeatures, selection_order
from reprforge.residual_materialization_oracle import (
    ResidualRankSurface,
    aggregate_runs,
    auc,
    direct_ranking_evaluation,
    evaluate,
    gain_recovery,
    global_label_rank_utility,
    hash_folds,
    omni_frequency_utility,
    projected_cost,
    residual_events,
    residual_repaired_fraction,
    residual_utility,
    reuse_diagnostics,
    top_utility,
)
from tools.run_dvi_page_verifier_pilot import (
    _bm25_rankings,
    _load_qrels,
    _load_visual_ranking,
    _read_jsonl,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _rank_matrix(
    rankings: dict[str, list[str]],
    query_ids: list[str],
    doc_positions: dict[str, int],
    depth: int,
    *,
    label: str,
) -> np.ndarray:
    rows = []
    for query_id in query_ids:
        if query_id not in rankings:
            raise ValueError(f"{label} lacks query {query_id}")
        docs = rankings[query_id][:depth]
        if len(docs) != depth:
            raise ValueError(f"{label} query {query_id} has only {len(docs)} pages")
        try:
            rows.append([doc_positions[doc_id] for doc_id in docs])
        except KeyError as error:
            raise ValueError(f"{label} contains unknown page {error.args[0]}") from error
    return np.asarray(rows, dtype=np.int32)


def _surface(
    *,
    name: str,
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, float]],
    bm25: dict[str, list[str]],
    colsmol: dict[str, list[str]],
    omni: dict[str, list[str]],
    depth: int,
) -> ResidualRankSurface:
    doc_ids = [str(row["docid"]) for row in corpus]
    query_ids = [str(row["query_id"]) for row in queries]
    doc_positions = {doc_id: position for position, doc_id in enumerate(doc_ids)}
    qrel_matrix = np.zeros((len(query_ids), len(doc_ids)), dtype=np.float32)
    for query_position, query_id in enumerate(query_ids):
        for doc_id, relevance in qrels[query_id].items():
            qrel_matrix[query_position, doc_positions[doc_id]] = float(relevance)
    return ResidualRankSurface(
        name=name,
        query_ids=query_ids,
        doc_ids=doc_ids,
        bm25=_rank_matrix(bm25, query_ids, doc_positions, depth, label="bm25"),
        colsmol=_rank_matrix(
            colsmol, query_ids, doc_positions, depth, label="colsmol"
        ),
        omni=_rank_matrix(omni, query_ids, doc_positions, depth, label="omni"),
        qrels=qrel_matrix,
    )


def _strip_rankings(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "rankings"}


def _annotate(
    row: dict[str, Any],
    *,
    surface: ResidualRankSurface,
    residual_queries: list[int],
    residual_depth: int,
    base_ndcg: float,
    full_ndcg: float,
    full_residual_repair: float,
    cost: dict[str, Any],
) -> dict[str, Any]:
    repaired = residual_repaired_fraction(
        surface,
        residual_queries,
        [row["rankings"][query] for query in residual_queries],
        depth=residual_depth,
    )
    selected_fraction = float(row["selected_page_fraction"])
    return {
        **_strip_rankings(row),
        "ndcg_gain_recovery": gain_recovery(
            float(row["ndcg_at_10"]), base_ndcg, full_ndcg
        ),
        "residual_repaired_fraction": repaired,
        "residual_repair_recovery_vs_full_stack": (
            repaired / full_residual_repair if full_residual_repair > 0 else None
        ),
        "projected_cost": projected_cost(
            selected_fraction,
            full_build_seconds=float(cost["full_omni_build_seconds"]),
            full_index_bytes=int(cost["full_omni_index_bytes"]),
            base_build_seconds=float(cost["base_colsmol_build_seconds"]),
            base_index_bytes=int(cost["base_colsmol_index_bytes"]),
        ),
    }


def _feature_diagnostics(
    surface: ResidualRankSurface,
    features_path: Path,
    residual_pages: set[int],
    budgets: list[float],
) -> tuple[dict[str, Any], list[int]]:
    rows = [json.loads(line) for line in features_path.read_text().splitlines() if line]
    by_id = {str(row["doc_id"]): row for row in rows}
    if set(by_id) != set(surface.doc_ids):
        raise ValueError("feature page IDs differ from corpus")
    features = [
        PageRiskFeatures(
            doc_id=doc_id,
            text_chars=int(by_id[doc_id]["text_chars"]),
            grayscale_entropy=float(by_id[doc_id]["grayscale_entropy"]),
            edge_energy=float(by_id[doc_id]["edge_energy"]),
            nonwhite_fraction=float(by_id[doc_id]["nonwhite_fraction"]),
        )
        for doc_id in surface.doc_ids
    ]
    positions = {doc_id: index for index, doc_id in enumerate(surface.doc_ids)}
    risk_order = [
        positions[doc_id]
        for doc_id in selection_order(
            features, strategy="risk_cover_round_robin", seed=20260807
        )
    ]
    labels = np.asarray([page in residual_pages for page in range(surface.pages)])
    values = {
        "text_scarcity": [-row.text_chars for row in features],
        "grayscale_entropy": [row.grayscale_entropy for row in features],
        "edge_energy": [row.edge_energy for row in features],
        "nonwhite_fraction": [row.nonwhite_fraction for row in features],
    }
    feature_rows = {}
    for name, raw_values in values.items():
        array = np.asarray(raw_values, dtype=np.float64)
        feature_rows[name] = {
            "auc_for_residual_page": auc(array, labels),
            "residual_mean": float(np.mean(array[labels])) if np.any(labels) else None,
            "other_mean": float(np.mean(array[~labels])) if np.any(~labels) else None,
            "residual_median": (
                float(np.median(array[labels])) if np.any(labels) else None
            ),
            "other_median": float(np.median(array[~labels])) if np.any(~labels) else None,
        }
    coverage = {}
    for budget in budgets:
        count = 0 if budget == 0 else min(surface.pages, max(1, math.ceil(budget * surface.pages)))
        selected = set(risk_order[:count])
        coverage[str(budget)] = {
            "selected_pages": count,
            "residual_page_coverage": (
                len(selected & residual_pages) / len(residual_pages)
                if residual_pages
                else None
            ),
        }
    return {
        "residual_pages": len(residual_pages),
        "feature_separation": feature_rows,
        "physical_risk_coverage": coverage,
    }, risk_order


def _natural_reuse(
    surface: ResidualRankSurface,
    *,
    split_fraction: float,
    rrf_constant: int,
    depth: int,
) -> dict[str, Any]:
    split = max(1, min(surface.queries - 1, int(surface.queries * split_fraction)))
    history = residual_events(
        surface, np.arange(split), rrf_constant=rrf_constant, depth=depth
    )
    future = residual_events(
        surface, np.arange(split, surface.queries), rrf_constant=rrf_constant, depth=depth
    )
    history_pages = history["unique_pages"]
    future_pages = future["unique_pages"]
    future_events = future["events"]
    return {
        "split_query_position": split,
        "history_residual_queries": history["queries"],
        "future_residual_queries": future["queries"],
        "future_unique_page_overlap_fraction": (
            len(history_pages & future_pages) / len(future_pages)
            if future_pages
            else None
        ),
        "future_event_overlap_fraction": (
            sum(len(pages & history_pages) for _, pages in future_events)
            / future["page_events"]
            if future["page_events"]
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--colsmol-ranking", type=Path, required=True)
    parser.add_argument("--omni-ranking", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    retrieval = config["retrieval"]
    depth = int(retrieval["rank_depth"])
    residual_depth = int(retrieval["residual_depth"])
    rrf_constant = int(retrieval["rrf_constant"])
    budgets = list(map(float, config["budgets"]))
    corpus = _read_jsonl(args.dataset_root / "corpus.jsonl")
    queries = _read_jsonl(args.dataset_root / "queries.jsonl")
    qrels = _load_qrels(args.dataset_root / "qrels.jsonl")
    bm25, bm25_cost = _bm25_rankings(corpus, queries, depth)
    colsmol = _load_visual_ranking(args.colsmol_ranking, depth)
    omni = _load_visual_ranking(args.omni_ranking, depth)
    surface = _surface(
        name=args.domain,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        bm25=bm25,
        colsmol=colsmol,
        omni=omni,
        depth=depth,
    )
    all_queries = np.arange(surface.queries)
    all_pages = np.arange(surface.pages)
    base = evaluate(
        surface,
        all_queries,
        rrf_constant=rrf_constant,
        selected_omni_pages=None,
    )
    full = evaluate(
        surface,
        all_queries,
        rrf_constant=rrf_constant,
        selected_omni_pages=all_pages,
    )
    bm25_only = direct_ranking_evaluation(surface, all_queries, surface.bm25)
    colsmol_only = direct_ranking_evaluation(surface, all_queries, surface.colsmol)
    omni_only = direct_ranking_evaluation(surface, all_queries, surface.omni)
    residual = residual_events(
        surface, all_queries, rrf_constant=rrf_constant, depth=residual_depth
    )
    residual_queries = [int(query) for query, _ in residual["events"]]
    full_residual_eval = evaluate(
        surface,
        residual_queries,
        rrf_constant=rrf_constant,
        selected_omni_pages=all_pages,
    )
    full_residual_repair = residual_repaired_fraction(
        surface,
        residual_queries,
        full_residual_eval["rankings"],
        depth=residual_depth,
    )
    cost = config["cost_accounting"][args.domain]
    base_ndcg = float(base["ndcg_at_10"])
    full_ndcg = float(full["ndcg_at_10"])
    residual_util = residual_utility(
        surface, all_queries, rrf_constant=rrf_constant, depth=residual_depth
    )
    global_util = global_label_rank_utility(
        surface, all_queries, rrf_constant=rrf_constant
    )
    frequency_util = omni_frequency_utility(surface, all_queries)
    feature_diagnostics, physical_order = _feature_diagnostics(
        surface, args.features, set(map(int, np.flatnonzero(residual_util > 0))), budgets
    )

    curves: dict[str, Any] = {
        policy: {}
        for policy in (
            "random",
            "corpus_uniform",
            "physical_risk",
            "omni_frequency_oracle",
            "residual_label_oracle",
            "global_label_rank_oracle",
        )
    }
    random_repetitions = int(config["random_repetitions"])
    seed = int(config["cross_fit"]["seed"])
    for budget in budgets:
        count = 0 if budget == 0 else min(surface.pages, max(1, math.ceil(budget * surface.pages)))
        selections: dict[str, list[np.ndarray]] = {
            "random": [
                np.sort(
                    np.random.default_rng(seed + repetition + count * 1000).choice(
                        surface.pages, count, replace=False
                    )
                )
                for repetition in range(random_repetitions)
            ],
            "corpus_uniform": [
                np.unique(np.linspace(0, surface.pages - 1, count, dtype=np.int32))
                if count
                else np.empty(0, dtype=np.int32)
            ],
            "physical_risk": [np.asarray(sorted(physical_order[:count]), dtype=np.int32)],
            "omni_frequency_oracle": [
                top_utility(frequency_util, count, positive_only=False)
            ],
            "residual_label_oracle": [
                top_utility(residual_util, count, positive_only=True)
            ],
            "global_label_rank_oracle": [
                top_utility(global_util, count, positive_only=True)
            ],
        }
        for policy, policy_selections in selections.items():
            rows = []
            for selected in policy_selections:
                row = evaluate(
                    surface,
                    all_queries,
                    rrf_constant=rrf_constant,
                    selected_omni_pages=selected,
                )
                rows.append(
                    _annotate(
                        row,
                        surface=surface,
                        residual_queries=residual_queries,
                        residual_depth=residual_depth,
                        base_ndcg=base_ndcg,
                        full_ndcg=full_ndcg,
                        full_residual_repair=full_residual_repair,
                        cost=cost,
                    )
                )
            if len(rows) == 1:
                curves[policy][str(budget)] = rows[0]
            else:
                aggregate = aggregate_runs(rows)
                mean_fraction = aggregate["selected_page_fraction"]["mean"]
                aggregate["ndcg_gain_recovery"] = gain_recovery(
                    aggregate["ndcg_at_10"]["mean"], base_ndcg, full_ndcg
                )
                aggregate["residual_repair_recovery_vs_full_stack"] = (
                    aggregate["residual_repaired_fraction"]["mean"]
                    / full_residual_repair
                    if full_residual_repair > 0
                    else None
                )
                aggregate["projected_cost_at_mean_fraction"] = projected_cost(
                    mean_fraction,
                    full_build_seconds=float(cost["full_omni_build_seconds"]),
                    full_index_bytes=int(cost["full_omni_index_bytes"]),
                    base_build_seconds=float(cost["base_colsmol_build_seconds"]),
                    base_index_bytes=int(cost["base_colsmol_index_bytes"]),
                )
                curves[policy][str(budget)] = aggregate

    folds = int(config["cross_fit"]["folds"])
    assignments = hash_folds(surface.query_ids, folds, seed)
    history_curves = {}
    for budget in budgets:
        count = 0 if budget == 0 else min(surface.pages, max(1, math.ceil(budget * surface.pages)))
        fold_rows = []
        for fold in range(folds):
            history = np.flatnonzero(assignments != fold)
            future = np.flatnonzero(assignments == fold)
            utility = residual_utility(
                surface, history, rrf_constant=rrf_constant, depth=residual_depth
            )
            selected = top_utility(utility, count, positive_only=True)
            row = evaluate(
                surface,
                future,
                rrf_constant=rrf_constant,
                selected_omni_pages=selected,
            )
            future_residual = residual_events(
                surface, future, rrf_constant=rrf_constant, depth=residual_depth
            )
            future_residual_queries = [int(query) for query, _ in future_residual["events"]]
            residual_eval = evaluate(
                surface,
                future_residual_queries,
                rrf_constant=rrf_constant,
                selected_omni_pages=selected,
            )
            fold_rows.append(
                {
                    **_strip_rankings(row),
                    "residual_repaired_fraction": residual_repaired_fraction(
                        surface,
                        future_residual_queries,
                        residual_eval["rankings"],
                        depth=residual_depth,
                    ),
                }
            )
        history_curves[str(budget)] = aggregate_runs(fold_rows)

    reuse = reuse_diagnostics(
        surface,
        rrf_constant=rrf_constant,
        depth=residual_depth,
        folds=folds,
        seed=seed,
    )
    gate_budget = float(config["gate"]["maximum_oracle_page_fraction"])
    eligible_budgets = [budget for budget in budgets if budget <= gate_budget]
    best_ndcg_recovery = max(
        (
            curves["global_label_rank_oracle"][str(budget)]["ndcg_gain_recovery"]
            for budget in eligible_budgets
            if curves["global_label_rank_oracle"][str(budget)]["ndcg_gain_recovery"]
            is not None
        ),
        default=None,
    )
    best_repair_recovery = max(
        (
            curves["residual_label_oracle"][str(budget)][
                "residual_repair_recovery_vs_full_stack"
            ]
            for budget in eligible_budgets
            if curves["residual_label_oracle"][str(budget)][
                "residual_repair_recovery_vs_full_stack"
            ]
            is not None
        ),
        default=None,
    )
    event_overlap = reuse["event_overlap_fraction_weighted"]
    checks = {
        "full_stack_absolute_ndcg_gain": (
            full_ndcg - base_ndcg
            >= float(config["gate"]["minimum_full_stack_absolute_ndcg_gain"])
        ),
        "oracle_ndcg_gain_recovery": (
            best_ndcg_recovery is not None
            and best_ndcg_recovery
            >= float(config["gate"]["minimum_oracle_ndcg_gain_recovery"])
        ),
        "oracle_residual_repair_recovery": (
            best_repair_recovery is not None
            and best_repair_recovery
            >= float(config["gate"]["minimum_residual_query_repair_recovery"])
        ),
        "crossfit_residual_event_overlap": (
            event_overlap is not None
            and event_overlap
            >= float(config["gate"]["minimum_crossfit_residual_event_overlap"])
        ),
    }
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": _canonical_sha(args.config),
        "status": "complete",
        "domain": args.domain,
        "dataset": {
            "pages": surface.pages,
            "queries": surface.queries,
            "sha256": {
                "corpus": _sha(args.dataset_root / "corpus.jsonl"),
                "queries": _sha(args.dataset_root / "queries.jsonl"),
                "qrels": _sha(args.dataset_root / "qrels.jsonl"),
                "colsmol_ranking": _sha(args.colsmol_ranking),
                "omni_ranking": _sha(args.omni_ranking),
                "features": _sha(args.features),
            },
        },
        "bm25_cost": bm25_cost,
        "baselines": {
            "bm25_only": bm25_only,
            "colsmol_only": colsmol_only,
            "omni_only": omni_only,
            "base_bm25_colsmol": _strip_rankings(base),
            "full_bm25_colsmol_omni": {
                **_strip_rankings(full),
                "ndcg_gain_over_base": full_ndcg - base_ndcg,
                "residual_repaired_fraction": full_residual_repair,
            },
        },
        "residual_boundary": {
            key: value
            for key, value in residual.items()
            if key not in ("events", "strict_events", "unique_pages", "strict_candidate_escape_unique_pages")
        }
        | {
            "unique_page_ids": [surface.doc_ids[page] for page in sorted(residual["unique_pages"])],
            "strict_candidate_escape_unique_page_ids": [
                surface.doc_ids[page]
                for page in sorted(residual["strict_candidate_escape_unique_pages"])
            ],
        },
        "curves": curves,
        "history_residual_crossfit_curves": history_curves,
        "reuse": {
            "hash_crossfit": reuse,
            "natural_half_split": _natural_reuse(
                surface,
                split_fraction=float(config["cross_fit"]["natural_split_fraction"]),
                rrf_constant=rrf_constant,
                depth=residual_depth,
            ),
        },
        "observable_feature_diagnostics": feature_diagnostics,
        "measured_cost_reference": cost,
        "gate": {
            "best_oracle_ndcg_gain_recovery_at_or_below_budget": best_ndcg_recovery,
            "best_residual_repair_recovery_at_or_below_budget": best_repair_recovery,
            "crossfit_residual_event_overlap": event_overlap,
            "checks": checks,
            "passes_domain_headroom_gate": all(checks.values()),
        },
        "warnings": [
            retrieval["partial_semantics"],
            config["information_boundary"]["warning"],
            "RRF rank surfaces measure retrieval localization and ranking, not final answer quality.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"domain": args.domain, "baselines": result["baselines"], "residual": result["residual_boundary"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
