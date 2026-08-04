#!/usr/bin/env python3
"""Cross-fit query-prototype residual witness indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_residual_token_witness import _hybrid_documents, _residual_tensor
from analyze_token_witness_index import _embedding_views, _fold, _score_surface
from reprforge.heterogeneity_atlas import query_metrics
from reprforge.heterogeneous_index import _load_shard
from reprforge.token_witness_compiler import (
    compile_token_witnesses,
    matched_random_witnesses,
)
from reprforge.workload_probe_compiler import fit_spherical_probes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-bank", type=Path, required=True)
    parser.add_argument("--pooled-bank", type=Path, required=True)
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--pooled-runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probe-counts", type=int, nargs="+", default=[32, 64])
    parser.add_argument(
        "--residual-epsilons", type=float, nargs="+", default=[0.05, 0.1, 0.15]
    )
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    args = parser.parse_args()

    item_ids, full_vectors, full_offsets = _load_shard(
        args.full_bank / "routes" / "image"
    )
    query_ids, query_vectors, query_offsets = _load_shard(
        args.full_bank / "queries"
    )
    pooled_item_ids, pooled_vectors, pooled_offsets = _load_shard(
        args.pooled_bank / "routes" / "image-pool-9"
    )
    if item_ids != pooled_item_ids:
        raise ValueError("full and pooled banks have different item order")
    full_runtime = np.load(args.full_runtime, allow_pickle=False)
    pooled_runtime = np.load(args.pooled_runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    if item_ids != [str(value) for value in full_runtime["corpus_ids"]]:
        raise ValueError("bank and runtime corpus IDs differ")
    if query_ids != [str(value) for value in full_runtime["query_ids"]]:
        raise ValueError("bank and runtime query IDs differ")

    queries = _embedding_views(query_vectors, query_offsets)
    full_documents = _embedding_views(full_vectors, full_offsets)
    pooled_documents = _embedding_views(pooled_vectors, pooled_offsets)
    full_counts = np.diff(full_offsets)
    pooled_counts = np.diff(pooled_offsets)
    relevance = [dict() for _ in query_ids]
    for query, corpus, value in zip(
        labels["query_positions"],
        labels["corpus_positions"],
        labels["relevance"],
        strict=True,
    ):
        relevance[int(query)][int(corpus)] = float(value)
    folds = np.asarray([_fold(value) for value in query_ids], dtype=np.int16)

    reports = {}
    for probe_count in args.probe_counts:
        states = {
            epsilon: {
                "witness": {
                    "ndcg_at_5": np.zeros(len(queries)),
                    "ndcg_at_10": np.zeros(len(queries)),
                    "recall_at_100": np.zeros(len(queries)),
                },
                "random": {
                    "ndcg_at_5": np.zeros(len(queries)),
                    "ndcg_at_10": np.zeros(len(queries)),
                    "recall_at_100": np.zeros(len(queries)),
                },
                "folds": [],
            }
            for epsilon in args.residual_epsilons
        }
        for fold in sorted(set(folds.tolist())):
            fit = np.flatnonzero(folds != fold)
            test = np.flatnonzero(folds == fold)
            fit_tokens = np.concatenate([queries[index] for index in fit], axis=0)
            probes, quantization_errors = fit_spherical_probes(
                fit_tokens,
                count=probe_count,
                seed=20260803 + fold + probe_count * 100,
                iterations=args.kmeans_iterations,
            )
            winners, residuals, residual_ms = _residual_tensor(
                (probes,),
                full_documents,
                pooled_documents,
                device=args.device,
                batch_size=args.batch_size,
            )
            for epsilon, state in states.items():
                eligible_winners = winners.copy()
                eligible_winners[residuals <= epsilon] = -1
                plan = compile_token_witnesses(
                    eligible_winners,
                    fit_queries=(0,),
                    document_token_counts=full_counts,
                    minimum_win_count=1,
                    minimum_tokens=0,
                )
                random_plan = matched_random_witnesses(
                    plan,
                    document_token_counts=full_counts,
                    seed=20260803 + fold + probe_count * 1000 + int(epsilon * 10000),
                )
                hybrid = _hybrid_documents(full_documents, pooled_documents, plan)
                random_hybrid = _hybrid_documents(
                    full_documents, pooled_documents, random_plan
                )
                hybrid_selections = tuple(
                    np.arange(len(value), dtype=np.int32) for value in hybrid
                )
                random_selections = tuple(
                    np.arange(len(value), dtype=np.int32) for value in random_hybrid
                )
                test_queries = tuple(queries[index] for index in test)
                scores, score_ms = _score_surface(
                    test_queries,
                    hybrid,
                    hybrid_selections,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                random_scores, random_ms = _score_surface(
                    test_queries,
                    random_hybrid,
                    random_selections,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                test_relevance = tuple(relevance[index] for index in test)
                metrics = query_metrics(scores, test_relevance, ks=(5, 10, 100))
                random_metrics = query_metrics(
                    random_scores, test_relevance, ks=(5, 10, 100)
                )
                for metric in state["witness"]:
                    state["witness"][metric][test] = metrics[metric]
                    state["random"][metric][test] = random_metrics[metric]
                residual_counts = np.asarray([len(value) for value in plan])
                total_tokens = int(pooled_counts.sum() + residual_counts.sum())
                state["folds"].append(
                    {
                        "fold": fold,
                        "fit_query_tokens": len(fit_tokens),
                        "probe_quantization_mean": float(quantization_errors.mean()),
                        "probe_quantization_p95": float(
                            np.quantile(quantization_errors, 0.95)
                        ),
                        "residual_tensor_ms": residual_ms,
                        "residual_tokens": int(residual_counts.sum()),
                        "documents_with_residual_fraction": float(
                            np.mean(residual_counts > 0)
                        ),
                        "median_residual_tokens": float(
                            np.median(residual_counts)
                        ),
                        "p95_residual_tokens": float(
                            np.quantile(residual_counts, 0.95)
                        ),
                        "maximum_residual_tokens": int(residual_counts.max()),
                        "token_fraction": total_tokens / int(full_counts.sum()),
                        "score_ms": score_ms,
                        "random_score_ms": random_ms,
                    }
                )
        for epsilon, state in states.items():
            key = f"p{probe_count}-eps{epsilon:g}"
            reports[key] = {
                "probe_count": probe_count,
                "residual_epsilon": epsilon,
                "mean_token_fraction": float(
                    np.mean([row["token_fraction"] for row in state["folds"]])
                ),
                "witness": {
                    name: float(values.mean())
                    for name, values in state["witness"].items()
                },
                "matched_random": {
                    name: float(values.mean())
                    for name, values in state["random"].items()
                },
                "per_query_witness": {
                    name: values.tolist()
                    for name, values in state["witness"].items()
                },
                "folds": state["folds"],
            }

    baselines = {}
    for name, scores in (
        ("full", np.asarray(full_runtime["scores"], dtype=np.float64)),
        ("pool9", np.asarray(pooled_runtime["scores"], dtype=np.float64)),
    ):
        values = query_metrics(scores, tuple(relevance), ks=(5, 10, 100))
        baselines[name] = {
            key: float(metric.mean())
            for key, metric in values.items()
            if key in {"ndcg_at_5", "ndcg_at_10", "recall_at_100"}
        }
    result = {
        "protocol": (
            "pool9 cover plus qrel-free spherical workload probes and "
            "full-token residual witnesses; query-hash five-fold crossfit"
        ),
        "compiler_uses_qrels": False,
        "pool9_token_fraction": float(pooled_counts.sum() / full_counts.sum()),
        "baselines": baselines,
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
