#!/usr/bin/env python3
"""Cross-fit homogeneous token selection guided by workload query prototypes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tools.analyze_token_witness_index import _embedding_views, _fold, _score_surface
from reprforge.heterogeneity_atlas import paired_bootstrap_ci, query_metrics
from reprforge.heterogeneous_index import _load_shard
from reprforge.workload_guided_token_compression import (
    workload_guided_token_selection,
)
from reprforge.workload_probe_compiler import fit_spherical_probes


def _relevance(labels, *, query_count: int, corpus_count: int):
    rows = [dict() for _ in range(query_count)]
    for query, corpus, value in zip(
        labels["query_positions"], labels["corpus_positions"], labels["relevance"], strict=True
    ):
        query_index = int(query)
        corpus_index = int(corpus)
        if not 0 <= query_index < query_count or not 0 <= corpus_index < corpus_count:
            raise ValueError("label position lies outside score surface")
        rows[query_index][corpus_index] = float(value)
    if any(not row for row in rows):
        raise ValueError("every query must have at least one relevance judgment")
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--pool-runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--budgets", type=int, nargs="+", default=(64, 128))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    item_ids, document_vectors, document_offsets = _load_shard(args.bank / "routes" / "image")
    query_ids, query_vectors, query_offsets = _load_shard(args.bank / "queries")
    full_runtime = np.load(args.full_runtime, allow_pickle=False)
    pool_runtime = np.load(args.pool_runtime, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    if item_ids != [str(value) for value in full_runtime["corpus_ids"]]:
        raise ValueError("bank and full runtime corpus IDs differ")
    if query_ids != [str(value) for value in full_runtime["query_ids"]]:
        raise ValueError("bank and full runtime query IDs differ")
    if not np.array_equal(full_runtime["query_ids"], pool_runtime["query_ids"]) or not np.array_equal(
        full_runtime["corpus_ids"], pool_runtime["corpus_ids"]
    ):
        raise ValueError("full and pooled runtimes are not aligned")
    queries = _embedding_views(query_vectors, query_offsets)
    documents = _embedding_views(document_vectors, document_offsets)
    document_counts = np.diff(document_offsets)
    relevance = _relevance(labels, query_count=len(queries), corpus_count=len(documents))
    fold_ids = np.asarray([_fold(value, count=args.folds) for value in query_ids])
    full_scores = np.asarray(full_runtime["scores"], dtype=np.float64)
    baselines = {
        "full": query_metrics(full_scores, relevance, ks=(5, 10, 100)),
        "pool": query_metrics(
            np.asarray(pool_runtime["scores"], dtype=np.float64), relevance, ks=(5, 10, 100)
        ),
    }
    metric_names = ("ndcg_at_5", "ndcg_at_10", "recall_at_100")
    predictions = {
        budget: {
            "workload": {name: np.zeros(len(queries)) for name in metric_names},
            "matched_random": {name: np.zeros(len(queries)) for name in metric_names},
        }
        for budget in args.budgets
    }
    fold_reports = []
    for fold in range(args.folds):
        fit = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        fit_tokens = np.concatenate([queries[index] for index in fit], axis=0)
        probes, errors = fit_spherical_probes(
            fit_tokens,
            count=args.probe_count,
            seed=args.seed + fold,
        )
        assignments = (fit_tokens @ probes.T).argmax(axis=1)
        weights = np.bincount(assignments, minlength=args.probe_count).astype(np.float64)
        for budget in args.budgets:
            began = time.perf_counter()
            selections = tuple(
                workload_guided_token_selection(
                    document,
                    probes,
                    budget=budget,
                    probe_weights=weights,
                )
                for document in documents
            )
            compile_ms = (time.perf_counter() - began) * 1000.0
            generator = np.random.default_rng(args.seed + fold * 1000 + budget)
            random_selections = tuple(
                np.asarray(
                    sorted(
                        generator.choice(
                            len(document), size=min(budget, len(document)), replace=False
                        ).tolist()
                    ),
                    dtype=np.int32,
                )
                for document in documents
            )
            test_queries = tuple(queries[index] for index in test)
            workload_scores, workload_ms = _score_surface(
                test_queries,
                documents,
                selections,
                device=args.device,
                batch_size=args.batch_size,
            )
            random_scores, random_ms = _score_surface(
                test_queries,
                documents,
                random_selections,
                device=args.device,
                batch_size=args.batch_size,
            )
            test_relevance = tuple(relevance[index] for index in test)
            workload_metrics = query_metrics(workload_scores, test_relevance, ks=(5, 10, 100))
            random_metrics = query_metrics(random_scores, test_relevance, ks=(5, 10, 100))
            for name in metric_names:
                predictions[budget]["workload"][name][test] = workload_metrics[name]
                predictions[budget]["matched_random"][name][test] = random_metrics[name]
            selected_tokens = sum(len(value) for value in selections)
            fold_reports.append(
                {
                    "fold": fold,
                    "budget": budget,
                    "fit_queries": int(len(fit)),
                    "test_queries": int(len(test)),
                    "mean_probe_cosine_error": float(errors.mean()),
                    "p95_probe_cosine_error": float(np.quantile(errors, 0.95)),
                    "selected_tokens": int(selected_tokens),
                    "token_fraction": float(selected_tokens / document_counts.sum()),
                    "compile_ms": compile_ms,
                    "score_ms": workload_ms,
                    "matched_random_score_ms": random_ms,
                }
            )
    results = {}
    for budget, methods in predictions.items():
        results[str(budget)] = {}
        for method, metrics in methods.items():
            results[str(budget)][method] = {
                "mean": {name: float(values.mean()) for name, values in metrics.items()},
                "vs_full": {
                    name: paired_bootstrap_ci(values, baselines["full"][name], seed=args.seed)
                    for name, values in metrics.items()
                },
                "vs_pool": {
                    name: paired_bootstrap_ci(values, baselines["pool"][name], seed=args.seed)
                    for name, values in metrics.items()
                },
                "per_query": {name: values.tolist() for name, values in metrics.items()},
            }
    report = {
        "schema_version": 1,
        "stage": "opened-domain-workload-guided-homogeneous-token-probe",
        "protocol": "qrel-free query-prototype weighted-fair original-token selection; query-hash crossfit",
        "compiler_uses_qrels": False,
        "score_property": "selected-token MaxSim is pointwise no greater than full MaxSim",
        "queries": len(queries),
        "documents": len(documents),
        "full_tokens": int(document_counts.sum()),
        "probe_count": args.probe_count,
        "baselines": {
            route: {name: float(values[name].mean()) for name in metric_names}
            for route, values in baselines.items()
        },
        "results": results,
        "folds": fold_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "baselines": report["baselines"],
                "results": {
                    budget: {
                        method: values["mean"] for method, values in methods.items()
                    }
                    for budget, methods in results.items()
                },
                "mean_token_fraction": {
                    str(budget): float(
                        np.mean(
                            [
                                row["token_fraction"]
                                for row in fold_reports
                                if row["budget"] == budget
                            ]
                        )
                    )
                    for budget in args.budgets
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
