#!/usr/bin/env python3
"""Evaluate the leakage-safe two-action materialization compiler v0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.materialization import (
    CostCatalog,
    PageSignals,
    PolicyConfig,
    compile_plan,
    load_frozen_split,
    replay_feature_policy,
)
from reprforge.partial_vlm_materialization import (
    evaluate_selection,
    evaluate_text_only,
    gain_recovery,
    select_pages,
)
from tools.evaluate_value_aware_materialization import load_exported_surface


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_ranking(path: Path, query_ids: Sequence[str], corpus_ids: Sequence[str], depth: int) -> np.ndarray:
    rows: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        query_id, doc_id, *_ = line.split("\t")
        values = rows.setdefault(str(query_id), [])
        if len(values) < depth:
            values.append(str(doc_id))
    if set(rows) != set(map(str, query_ids)):
        raise ValueError("visual ranking query IDs differ from score surface")
    positions = {str(doc_id): index for index, doc_id in enumerate(corpus_ids)}
    output = []
    for query_id in map(str, query_ids):
        values = rows[query_id]
        if len(values) != depth or len(set(values)) != depth:
            raise ValueError("visual ranking depth is incomplete or contains duplicates")
        output.append([positions[doc_id] for doc_id in values])
    return np.asarray(output, dtype=np.int32)


def _load_page_features(path: Path, corpus_ids: Sequence[str]) -> dict[str, np.ndarray]:
    rows = _read_jsonl(path)
    by_id = {str(row["doc_id"]): row for row in rows}
    if set(by_id) != set(map(str, corpus_ids)):
        raise ValueError("page feature IDs differ from score surface")
    return {
        "text_chars": np.asarray(
            [float(by_id[str(doc_id)]["text_chars"]) for doc_id in corpus_ids]
        ),
        "grayscale_entropy": np.asarray(
            [float(by_id[str(doc_id)]["grayscale_entropy"]) for doc_id in corpus_ids]
        ),
        "edge_energy": np.asarray(
            [float(by_id[str(doc_id)]["edge_energy"]) for doc_id in corpus_ids]
        ),
    }


def _fit_signals(
    candidates: np.ndarray,
    text_order: np.ndarray,
    fit_queries: Sequence[int],
    page_features: dict[str, np.ndarray],
) -> PageSignals:
    pages = candidates.max(initial=-1) + 1
    pages = max(pages, text_order.shape[1])
    events = np.zeros(pages, dtype=np.float64)
    disagreement = np.zeros(pages, dtype=np.float64)
    for query in map(int, fit_queries):
        events[candidates[query]] += 1.0
        text = set(map(int, text_order[query, : candidates.shape[1]]))
        visual = set(map(int, candidates[query]))
        for page in text ^ visual:
            disagreement[page] += 1.0
    disagreement /= float(len(fit_queries))
    return PageSignals(
        page_ids=np.arange(pages, dtype=np.int32),
        fit_candidate_events=events,
        text_chars=page_features["text_chars"],
        grayscale_entropy=page_features["grayscale_entropy"],
        edge_energy=page_features["edge_energy"],
        locator_disagreement=disagreement,
    )


def _load_costs(feature_cache: Path, retrieval_construction: Path) -> CostCatalog:
    feature = json.loads(feature_cache.read_text())
    retrieval = json.loads(retrieval_construction.read_text())
    disk = feature["disk_cache"]
    batch_one = retrieval["results"]["1"]
    return CostCatalog(
        raw_query_seconds=float(feature["full_end_to_end_ms"]["mean"]) / 1000.0,
        feature_query_seconds=float(disk["read_h2d_and_language_ms"]["mean"]) / 1000.0,
        feature_build_seconds=float(feature["cache_build_end_to_end_ms"]["mean"]) / 1000.0,
        feature_write_seconds=float(disk["write_and_fsync_ms"]["mean"]) / 1000.0,
        feature_bytes=float(disk["mean_serialized_feature_bytes"]),
        retrieval_build_seconds=float(batch_one["end_to_end_ms_per_page"]["mean"]) / 1000.0,
        retrieval_bytes=float(batch_one["output_vector_bytes_per_page"]),
    )


def _top(values: np.ndarray, count: int) -> np.ndarray:
    pages = np.arange(len(values), dtype=np.int32)
    return pages[np.lexsort((pages, -np.asarray(values, dtype=np.float64)))[:count]]


def _compact_replay(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key != "cumulative_seconds"}


def _feature_baselines(
    candidates: np.ndarray,
    signals: PageSignals,
    plan_feature_pages: Sequence[int],
    test_order: np.ndarray,
    costs: CostCatalog,
    *,
    capacity_pages: int,
) -> dict[str, dict[str, object]]:
    future_events = np.zeros(len(signals.page_ids), dtype=np.float64)
    for query in map(int, test_order):
        future_events[candidates[query]] += 1.0
    history_pages = _top(signals.fit_candidate_events, capacity_pages)
    oracle_pages = _top(future_events, capacity_pages)
    rows = {
        "never_materialize": replay_feature_policy(
            candidates, test_order, costs, capacity_pages=0, policy="never"
        ),
        "full_materialize": replay_feature_policy(
            candidates,
            test_order,
            costs,
            capacity_pages=len(signals.page_ids),
            policy="static",
            initial_pages=np.arange(len(signals.page_ids)),
        ),
        "first_touch": replay_feature_policy(
            candidates,
            test_order,
            costs,
            capacity_pages=capacity_pages,
            policy="first_touch",
        ),
        "second_touch": replay_feature_policy(
            candidates,
            test_order,
            costs,
            capacity_pages=capacity_pages,
            policy="second_touch",
        ),
        "history_frequency": replay_feature_policy(
            candidates,
            test_order,
            costs,
            capacity_pages=capacity_pages,
            policy="static",
            initial_pages=history_pages,
        ),
        "compiler_reuse_value": replay_feature_policy(
            candidates,
            test_order,
            costs,
            capacity_pages=capacity_pages,
            policy="static",
            initial_pages=plan_feature_pages,
        ),
        "future_frequency_oracle": replay_feature_policy(
            candidates,
            test_order,
            costs,
            capacity_pages=capacity_pages,
            policy="static",
            initial_pages=oracle_pages,
        ),
    }
    return {name: _compact_replay(row) for name, row in rows.items()}


def _retrieval_quality(
    surface,
    split,
    signals: PageSignals,
    plan_retrieval_pages: Sequence[int],
    *,
    budget_pages: int,
    seed: int,
) -> dict[str, Any]:
    test = np.asarray(split.test, dtype=np.int32)
    text = _compact_quality(evaluate_text_only(surface, test))
    all_pages = np.arange(surface.pages, dtype=np.int32)
    fusion_protocols = {
        "naive_sparse_rrf100": {"fusion": "rrf", "visual_top_k": 100},
        "support_limited_rrf5": {"fusion": "rrf", "visual_top_k": 5},
        "missing_as_mean_prior_zscore": {"fusion": "zscore"},
    }
    full = {
        name: _compact_quality(
            evaluate_selection(surface, test, all_pages, **parameters)
        )
        for name, parameters in fusion_protocols.items()
    }
    risk_only_score = (
        _standardize(signals.locator_disagreement)
        - 0.5 * _standardize(np.log1p(signals.text_chars))
        + 0.125
        * (
            _standardize(signals.grayscale_entropy)
            + _standardize(signals.edge_energy)
        )
    )
    selections = {
        "random": np.random.default_rng(seed).permutation(surface.pages)[:budget_pages],
        "history_frequency": _top(signals.fit_candidate_events, budget_pages),
        "visual_risk_only": _top(risk_only_score, budget_pages),
        "compiler_joint_risk_reuse": np.asarray(plan_retrieval_pages, dtype=np.int32),
        "future_label_rank_oracle": select_pages(
            surface,
            policy="label_rank_oracle",
            count=budget_pages,
            history_queries=split.fit,
            future_queries=split.test,
            seed=seed,
        ),
    }
    rows: dict[str, Any] = {}
    for name, selected in selections.items():
        rows[name] = {}
        for fusion_name, parameters in fusion_protocols.items():
            quality = _compact_quality(
                evaluate_selection(surface, test, selected, **parameters)
            )
            rows[name][fusion_name] = {
                **quality,
                **gain_recovery(
                    quality["mean_ndcg_at_10"],
                    text["mean_ndcg_at_10"],
                    full[fusion_name]["mean_ndcg_at_10"],
                ),
            }
    return {
        "text_only": text,
        "full_hybrid": full,
        "budget_pages": budget_pages,
        "policies": rows,
        "warning": (
            "Sparse RRF ranks only the actually materialized support and can "
            "promote irrelevant pages. The mean-prior z-score path is qrel-free "
            "but is an audit protocol, not yet a certified production fusion. "
            "The label-rank policy is evaluation-only."
        ),
    }


def _compact_quality(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("per_query_")
    }


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    deviation = float(values.std())
    return np.zeros_like(values) if deviation <= 1e-12 else (values - values.mean()) / deviation


def run(args: argparse.Namespace) -> dict[str, Any]:
    surface, _, _ = load_exported_surface(args.score_root, args.dataset_root)
    split = load_frozen_split(args.split, list(map(str, surface.query_ids)))
    candidates = _load_ranking(
        args.visual_ranking,
        list(map(str, surface.query_ids)),
        list(map(str, surface.corpus_ids)),
        args.candidate_depth,
    )
    page_features = _load_page_features(args.features, surface.corpus_ids)
    signals = _fit_signals(candidates, surface.text_order, split.fit, page_features)
    costs = _load_costs(args.feature_cache, args.retrieval_construction)
    feature_budget_pages = int(math.ceil(args.feature_budget_fraction * surface.pages))
    orders = split.test_orders(range(args.order_seed, args.order_seed + args.random_orders))

    horizons = {}
    for repeats in args.trace_repeats:
        plan, diagnostics = compile_plan(
            signals,
            costs,
            fit_queries=len(split.fit),
            horizon_queries=len(split.test) * int(repeats),
            config=PolicyConfig(
                feature_budget_fraction=args.feature_budget_fraction,
                retrieval_budget_fraction=max(args.retrieval_budget_fractions),
            ),
        )
        order_rows = {}
        for name, order in orders.items():
            repeated = np.tile(order, int(repeats))
            order_rows[name] = _feature_baselines(
                candidates,
                signals,
                plan.feature_pages,
                repeated,
                costs,
                capacity_pages=feature_budget_pages,
            )
        horizons[str(repeats)] = {
            "trace_queries": len(split.test) * int(repeats),
            "compiler_feature_pages": len(plan.feature_pages),
            "compiler_expected_feature_net_seconds": plan.expected_feature_net_seconds,
            "predicted_positive_feature_pages": int(
                np.sum(diagnostics["feature_net_seconds"] > 0.0)
            ),
            "orders": order_rows,
        }

    retrieval_curves = {}
    for fraction in args.retrieval_budget_fractions:
        budget_plan, _ = compile_plan(
            signals,
            costs,
            fit_queries=len(split.fit),
            horizon_queries=len(split.test) * max(args.trace_repeats),
            config=PolicyConfig(
                feature_budget_fraction=args.feature_budget_fraction,
                retrieval_budget_fraction=fraction,
            ),
        )
        retrieval_curves[f"{fraction:.6g}"] = _retrieval_quality(
            surface,
            split,
            signals,
            budget_plan.retrieval_pages,
            budget_pages=int(math.ceil(fraction * surface.pages)),
            seed=args.order_seed,
        )
    return {
        "schema_version": 1,
        "protocol": "reprforge-two-action-materialization-compiler-v0",
        "domain": surface.name,
        "pages": surface.pages,
        "queries": surface.queries,
        "split": {
            "fit_queries": len(split.fit),
            "calibration_queries": len(split.calibration),
            "test_queries": len(split.test),
            "fit_folds": [1, 2, 3],
            "calibration_folds": [4],
            "test_fold": 0,
            "assignment": split.assignment,
            "seed": split.seed,
        },
        "information_boundary": {
            "compiler_uses_test_qrels": False,
            "compiler_uses_test_visual_scores": False,
            "compiler_uses_calibration_for_v0_weights": False,
            "retrieval_oracle_uses_test_labels": True,
        },
        "cost_catalog": costs.__dict__,
        "feature_budget_fraction": args.feature_budget_fraction,
        "retrieval_budget_fractions": args.retrieval_budget_fractions,
        "feature_break_even_future_uses": costs.feature_break_even_future_uses,
        "feature_replay": horizons,
        "retrieval_quality_by_budget": retrieval_curves,
        "input_sha256": {
            "score_manifest": _sha(args.score_root / "manifest.json"),
            "corpus": _sha(args.dataset_root / "corpus.jsonl"),
            "queries": _sha(args.dataset_root / "queries.jsonl"),
            "qrels": _sha(args.dataset_root / "qrels.jsonl"),
            "features": _sha(args.features),
            "split": _sha(args.split),
            "visual_ranking": _sha(args.visual_ranking),
            "feature_cache": _sha(args.feature_cache),
            "retrieval_construction": _sha(args.retrieval_construction),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--visual-ranking", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--retrieval-construction", type=Path, required=True)
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--feature-budget-fraction", type=float, default=0.05)
    parser.add_argument(
        "--retrieval-budget-fractions",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0],
    )
    parser.add_argument("--trace-repeats", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--random-orders", type=int, default=5)
    parser.add_argument("--order-seed", type=int, default=20260808)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "domain": result["domain"],
                "output": str(args.output),
                "pages": result["pages"],
                "split": result["split"],
                "feature_break_even_future_uses": result[
                    "feature_break_even_future_uses"
                ],
                "retrieval_budget_fractions": result[
                    "retrieval_budget_fractions"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
