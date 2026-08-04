#!/usr/bin/env python3
"""Cross-fit listwise competitive token witnesses on a ViDoRe bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_token_witness_index import (
    _embedding_views,
    _fold,
    _score_surface,
    _winner_tensor,
)
from reprforge.heterogeneity_atlas import query_metrics
from reprforge.heterogeneous_index import _load_shard
from reprforge.token_witness_compiler import (
    compile_token_witnesses,
    matched_random_witnesses,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-ks", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--minimum-win-counts", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--minimum-tokens", type=int, default=8)
    args = parser.parse_args()
    item_ids, document_vectors, document_offsets = _load_shard(
        args.bank / "routes" / "image"
    )
    query_ids, query_vectors, query_offsets = _load_shard(args.bank / "queries")
    runtime = np.load(args.runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    if item_ids != [str(value) for value in runtime["corpus_ids"]]:
        raise ValueError("bank and runtime corpus IDs differ")
    if query_ids != [str(value) for value in runtime["query_ids"]]:
        raise ValueError("bank and runtime query IDs differ")
    queries = _embedding_views(query_vectors, query_offsets)
    documents = _embedding_views(document_vectors, document_offsets)
    document_counts = np.diff(document_offsets)
    relevance = [dict() for _ in query_ids]
    for query, corpus, value in zip(
        labels["query_positions"],
        labels["corpus_positions"],
        labels["relevance"],
        strict=True,
    ):
        relevance[int(query)][int(corpus)] = float(value)
    folds = np.asarray([_fold(value) for value in query_ids], dtype=np.int16)
    winners, winner_ms = _winner_tensor(
        queries, documents, device=args.device, batch_size=args.batch_size
    )
    positions = np.arange(len(documents))
    full_scores = np.asarray(runtime["scores"], dtype=np.float64)
    full_order = np.stack(
        [np.lexsort((positions, -row)) for row in full_scores]
    )
    reports = {}
    for candidate_k in args.candidate_ks:
        pair_mask = np.zeros(full_scores.shape, dtype=bool)
        pair_mask[
            np.arange(len(queries))[:, None], full_order[:, :candidate_k]
        ] = True
        for threshold in args.minimum_win_counts:
            witness_predictions = {
                "ndcg_at_5": np.zeros(len(queries)),
                "ndcg_at_10": np.zeros(len(queries)),
                "recall_at_100": np.zeros(len(queries)),
            }
            random_predictions = {
                key: np.zeros(len(queries)) for key in witness_predictions
            }
            fold_rows = []
            for fold in sorted(set(folds.tolist())):
                fit = np.flatnonzero(folds != fold)
                test = np.flatnonzero(folds == fold)
                plan = compile_token_witnesses(
                    winners,
                    fit_queries=fit,
                    document_token_counts=document_counts,
                    minimum_win_count=threshold,
                    minimum_tokens=args.minimum_tokens,
                    competitive_pairs=pair_mask,
                )
                random_plan = matched_random_witnesses(
                    plan,
                    document_token_counts=document_counts,
                    seed=20260803 + fold + 1000 * candidate_k + threshold,
                )
                plan_scores, plan_ms = _score_surface(
                    tuple(queries[index] for index in test),
                    documents,
                    plan,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                random_scores, random_ms = _score_surface(
                    tuple(queries[index] for index in test),
                    documents,
                    random_plan,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                test_relevance = tuple(relevance[index] for index in test)
                plan_metrics = query_metrics(
                    plan_scores, test_relevance, ks=(5, 10, 100)
                )
                random_metrics = query_metrics(
                    random_scores, test_relevance, ks=(5, 10, 100)
                )
                for key in witness_predictions:
                    witness_predictions[key][test] = plan_metrics[key]
                    random_predictions[key][test] = random_metrics[key]
                selected = np.asarray([len(value) for value in plan])
                fold_rows.append(
                    {
                        "fold": fold,
                        "token_fraction": float(selected.sum() / document_counts.sum()),
                        "min_tokens_per_document": int(selected.min()),
                        "median_tokens_per_document": float(np.median(selected)),
                        "p95_tokens_per_document": float(np.quantile(selected, 0.95)),
                        "max_tokens_per_document": int(selected.max()),
                        "witness_score_ms": plan_ms,
                        "random_score_ms": random_ms,
                    }
                )
            reports[f"k{candidate_k}-w{threshold}"] = {
                "candidate_k": candidate_k,
                "minimum_win_count": threshold,
                "mean_token_fraction": float(
                    np.mean([row["token_fraction"] for row in fold_rows])
                ),
                "witness": {
                    key: float(values.mean())
                    for key, values in witness_predictions.items()
                },
                "matched_random": {
                    key: float(values.mean())
                    for key, values in random_predictions.items()
                },
                "per_query_witness": {
                    key: values.tolist() for key, values in witness_predictions.items()
                },
                "folds": fold_rows,
            }
    full_metrics = query_metrics(full_scores, tuple(relevance), ks=(5, 10, 100))
    result = {
        "protocol": "full-score Top-K competitive pairs; query-hash five-fold token-witness crossfit",
        "compiler_uses_qrels": False,
        "winner_tensor_ms": winner_ms,
        "full_trace": {
            key: float(values.mean())
            for key, values in full_metrics.items()
            if key in {"ndcg_at_5", "ndcg_at_10", "recall_at_100"}
        },
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: {
                    "token_fraction": value["mean_token_fraction"],
                    "witness": value["witness"],
                    "matched_random": value["matched_random"],
                }
                for key, value in reports.items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
