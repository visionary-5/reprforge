#!/usr/bin/env python3
"""Cross-fit pooled-cover plus full-token residual witness indexes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analyze_token_witness_index import _embedding_views, _fold, _score_surface
from reprforge.heterogeneity_atlas import query_metrics
from reprforge.heterogeneous_index import _load_shard
from reprforge.token_witness_compiler import (
    compile_token_witnesses,
    matched_random_witnesses,
)


def _residual_tensor(
    queries,
    full_documents,
    pooled_documents,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    import torch

    torch_device = torch.device(device)
    maximum_query_tokens = max(len(value) for value in queries)
    winners = np.full(
        (len(queries), len(full_documents), maximum_query_tokens),
        -1,
        dtype=np.int16,
    )
    residuals = np.full(winners.shape, -np.inf, dtype=np.float16)
    began = time.perf_counter()
    for document_start in range(0, len(full_documents), batch_size):
        full_values = full_documents[document_start : document_start + batch_size]
        pooled_values = pooled_documents[
            document_start : document_start + batch_size
        ]
        full_lengths = torch.tensor(
            [len(value) for value in full_values], device=torch_device
        )
        pooled_lengths = torch.tensor(
            [len(value) for value in pooled_values], device=torch_device
        )
        padded_full = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(np.asarray(value), dtype=torch.float32) for value in full_values],
            batch_first=True,
        ).to(torch_device)
        padded_pooled = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(np.asarray(value), dtype=torch.float32) for value in pooled_values],
            batch_first=True,
        ).to(torch_device)
        full_positions = torch.arange(padded_full.shape[1], device=torch_device)
        pooled_positions = torch.arange(
            padded_pooled.shape[1], device=torch_device
        )
        for query_start in range(0, len(queries), batch_size):
            query_values = queries[query_start : query_start + batch_size]
            padded_queries = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(np.asarray(value), dtype=torch.float32) for value in query_values],
                batch_first=True,
            ).to(torch_device)
            with torch.inference_mode():
                full_similarities = torch.einsum(
                    "aqd,bkd->abqk", padded_queries, padded_full
                ).masked_fill(
                    full_positions[None, None, None, :]
                    >= full_lengths[None, :, None, None],
                    float("-inf"),
                )
                pooled_similarities = torch.einsum(
                    "aqd,bkd->abqk", padded_queries, padded_pooled
                ).masked_fill(
                    pooled_positions[None, None, None, :]
                    >= pooled_lengths[None, :, None, None],
                    float("-inf"),
                )
                full_max, full_indices = full_similarities.max(dim=-1)
                pooled_max = pooled_similarities.max(dim=-1).values
                difference = full_max - pooled_max
            index_values = full_indices.cpu().numpy()
            residual_values = difference.cpu().numpy()
            for query_offset, query in enumerate(query_values):
                target = (
                    query_start + query_offset,
                    slice(document_start, document_start + len(full_values)),
                    slice(0, len(query)),
                )
                winners[target] = index_values[query_offset, :, : len(query)]
                residuals[target] = residual_values[
                    query_offset, :, : len(query)
                ].astype(np.float16)
    torch.cuda.synchronize(torch_device)
    return winners, residuals, (time.perf_counter() - began) * 1000.0


def _hybrid_documents(full_documents, pooled_documents, plan):
    return tuple(
        np.concatenate(
            [np.asarray(pooled), np.asarray(full)[np.asarray(selected, dtype=np.int32)]],
            axis=0,
        )
        for full, pooled, selected in zip(
            full_documents, pooled_documents, plan, strict=True
        )
    )


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
    parser.add_argument("--residual-epsilons", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1])
    parser.add_argument("--candidate-ks", type=int, nargs="+", default=[0])
    parser.add_argument("--minimum-win-count", type=int, default=1)
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
        labels["query_positions"], labels["corpus_positions"], labels["relevance"], strict=True
    ):
        relevance[int(query)][int(corpus)] = float(value)
    folds = np.asarray([_fold(value) for value in query_ids], dtype=np.int16)
    winners, residuals, residual_ms = _residual_tensor(
        queries,
        full_documents,
        pooled_documents,
        device=args.device,
        batch_size=args.batch_size,
    )
    positions = np.arange(len(full_documents))
    full_scores = np.asarray(full_runtime["scores"], dtype=np.float64)
    full_order = np.stack([np.lexsort((positions, -row)) for row in full_scores])
    reports = {}
    for candidate_k in args.candidate_ks:
        pair_mask = None
        if candidate_k > 0:
            pair_mask = np.zeros(full_scores.shape, dtype=bool)
            pair_mask[
                np.arange(len(queries))[:, None], full_order[:, :candidate_k]
            ] = True
        for epsilon in args.residual_epsilons:
            eligible_winners = winners.copy()
            eligible_winners[residuals <= epsilon] = -1
            witness_predictions = {
                "ndcg_at_5": np.zeros(len(queries)),
                "ndcg_at_10": np.zeros(len(queries)),
                "recall_at_100": np.zeros(len(queries)),
            }
            random_predictions = {key: np.zeros(len(queries)) for key in witness_predictions}
            fold_rows = []
            for fold in sorted(set(folds.tolist())):
                fit = np.flatnonzero(folds != fold)
                test = np.flatnonzero(folds == fold)
                plan = compile_token_witnesses(
                    eligible_winners,
                    fit_queries=fit,
                    document_token_counts=full_counts,
                    minimum_win_count=args.minimum_win_count,
                    minimum_tokens=0,
                    competitive_pairs=pair_mask,
                )
                random_plan = matched_random_witnesses(
                    plan,
                    document_token_counts=full_counts,
                    seed=20260803 + fold + max(candidate_k, 0) * 1000 + int(epsilon * 10000),
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
                scores, score_ms = _score_surface(
                    tuple(queries[index] for index in test),
                    hybrid,
                    hybrid_selections,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                random_scores, random_ms = _score_surface(
                    tuple(queries[index] for index in test),
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
                for key in witness_predictions:
                    witness_predictions[key][test] = metrics[key]
                    random_predictions[key][test] = random_metrics[key]
                residual_counts = np.asarray([len(value) for value in plan])
                total_tokens = int(pooled_counts.sum() + residual_counts.sum())
                fold_rows.append(
                    {
                        "fold": fold,
                        "token_fraction": total_tokens / int(full_counts.sum()),
                        "residual_tokens": int(residual_counts.sum()),
                        "documents_with_residual_fraction": float(np.mean(residual_counts > 0)),
                        "median_residual_tokens": float(np.median(residual_counts)),
                        "p95_residual_tokens": float(np.quantile(residual_counts, 0.95)),
                        "score_ms": score_ms,
                        "random_score_ms": random_ms,
                    }
                )
            key = f"k{candidate_k}-eps{epsilon:g}"
            reports[key] = {
                "candidate_k": candidate_k,
                "residual_epsilon": epsilon,
                "mean_token_fraction": float(np.mean([row["token_fraction"] for row in fold_rows])),
                "witness": {name: float(values.mean()) for name, values in witness_predictions.items()},
                "matched_random": {name: float(values.mean()) for name, values in random_predictions.items()},
                "per_query_witness": {name: values.tolist() for name, values in witness_predictions.items()},
                "folds": fold_rows,
            }
    baselines = {}
    for name, scores in (("full", full_scores), ("pool9", np.asarray(pooled_runtime["scores"], dtype=np.float64))):
        values = query_metrics(scores, tuple(relevance), ks=(5, 10, 100))
        baselines[name] = {key: float(metric.mean()) for key, metric in values.items() if key in {"ndcg_at_5", "ndcg_at_10", "recall_at_100"}}
    result = {
        "protocol": "pool9 cover plus workload-conditioned full-token residual witnesses; query-hash five-fold crossfit",
        "compiler_uses_qrels": False,
        "residual_tensor_ms": residual_ms,
        "pool9_token_fraction": float(pooled_counts.sum() / full_counts.sum()),
        "baselines": baselines,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: {"token_fraction": value["mean_token_fraction"], "witness": value["witness"], "matched_random": value["matched_random"]} for key, value in reports.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
