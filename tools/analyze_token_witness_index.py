#!/usr/bin/env python3
"""Cross-fit workload-conditioned token witnesses on a ViDoRe bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from reprforge.heterogeneity_atlas import query_metrics
from reprforge.heterogeneous_index import _load_shard
from reprforge.token_witness_compiler import (
    compile_token_witnesses,
    matched_random_witnesses,
)


def _fold(value: str, count: int = 5) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % count


def _embedding_views(vectors: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(
        vectors[int(offsets[index]) : int(offsets[index + 1])]
        for index in range(len(offsets) - 1)
    )


def _winner_tensor(queries, documents, *, device: str, batch_size: int) -> tuple[np.ndarray, float]:
    import torch

    torch_device = torch.device(device)
    maximum_query_tokens = max(len(value) for value in queries)
    winners = np.full(
        (len(queries), len(documents), maximum_query_tokens),
        -1,
        dtype=np.int16,
    )
    began = time.perf_counter()
    for document_start in range(0, len(documents), batch_size):
        document_values = documents[document_start : document_start + batch_size]
        document_lengths = torch.tensor(
            [len(value) for value in document_values], device=torch_device
        )
        padded_documents = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(np.asarray(value), dtype=torch.float32) for value in document_values],
            batch_first=True,
        ).to(torch_device)
        document_positions = torch.arange(
            padded_documents.shape[1], device=torch_device
        )
        for query_start in range(0, len(queries), batch_size):
            query_values = queries[query_start : query_start + batch_size]
            padded_queries = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(np.asarray(value), dtype=torch.float32) for value in query_values],
                batch_first=True,
            ).to(torch_device)
            with torch.inference_mode():
                similarities = torch.einsum(
                    "aqd,bkd->abqk", padded_queries, padded_documents
                )
                similarities = similarities.masked_fill(
                    document_positions[None, None, None, :]
                    >= document_lengths[None, :, None, None],
                    float("-inf"),
                )
                indices = similarities.argmax(dim=-1).cpu().numpy()
            for query_offset, query in enumerate(query_values):
                winners[
                    query_start + query_offset,
                    document_start : document_start + len(document_values),
                    : len(query),
                ] = indices[query_offset, :, : len(query)]
    torch.cuda.synchronize(torch_device)
    return winners, (time.perf_counter() - began) * 1000.0


def _score_surface(
    queries,
    documents,
    selections,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    import torch

    torch_device = torch.device(device)
    scores = np.zeros((len(queries), len(documents)), dtype=np.float32)
    began = time.perf_counter()
    for document_start in range(0, len(documents), batch_size):
        selected_documents = [
            np.asarray(document)[np.asarray(selection, dtype=np.int32)]
            for document, selection in zip(
                documents[document_start : document_start + batch_size],
                selections[document_start : document_start + batch_size],
                strict=True,
            )
        ]
        document_lengths = torch.tensor(
            [len(value) for value in selected_documents], device=torch_device
        )
        padded_documents = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(np.asarray(value), dtype=torch.float32) for value in selected_documents],
            batch_first=True,
        ).to(torch_device)
        document_positions = torch.arange(
            padded_documents.shape[1], device=torch_device
        )
        for query_start in range(0, len(queries), batch_size):
            query_values = queries[query_start : query_start + batch_size]
            query_lengths = torch.tensor(
                [len(value) for value in query_values], device=torch_device
            )
            padded_queries = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(np.asarray(value), dtype=torch.float32) for value in query_values],
                batch_first=True,
            ).to(torch_device)
            query_positions = torch.arange(
                padded_queries.shape[1], device=torch_device
            )
            with torch.inference_mode():
                similarities = torch.einsum(
                    "aqd,bkd->abqk", padded_queries, padded_documents
                )
                similarities = similarities.masked_fill(
                    document_positions[None, None, None, :]
                    >= document_lengths[None, :, None, None],
                    float("-inf"),
                )
                maxsim = similarities.max(dim=-1).values
                query_mask = query_positions[None, :] < query_lengths[:, None]
                values = (maxsim * query_mask[:, None, :]).sum(dim=-1)
            scores[
                query_start : query_start + len(query_values),
                document_start : document_start + len(selected_documents),
            ] = values.cpu().numpy()
    torch.cuda.synchronize(torch_device)
    return scores, (time.perf_counter() - began) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--minimum-win-counts", type=int, nargs="+", default=[1, 2, 4, 8]
    )
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
        labels["query_positions"], labels["corpus_positions"], labels["relevance"], strict=True
    ):
        relevance[int(query)][int(corpus)] = float(value)
    folds = np.asarray([_fold(value) for value in query_ids], dtype=np.int16)
    full_metrics = query_metrics(
        np.asarray(runtime["scores"], dtype=np.float64),
        tuple(relevance),
        ks=(5, 10, 100),
    )
    winners, winner_ms = _winner_tensor(
        queries, documents, device=args.device, batch_size=args.batch_size
    )
    reports = {}
    for threshold in args.minimum_win_counts:
        witness_predictions = {"ndcg_at_5": np.zeros(len(queries)), "ndcg_at_10": np.zeros(len(queries)), "recall_at_100": np.zeros(len(queries))}
        random_predictions = {key: np.zeros(len(queries)) for key in witness_predictions}
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
            )
            random_plan = matched_random_witnesses(
                plan, document_token_counts=document_counts, seed=20260803 + fold
            )
            witness_scores, witness_ms = _score_surface(
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
            witness_metrics = query_metrics(witness_scores, test_relevance, ks=(5, 10, 100))
            random_metrics = query_metrics(random_scores, test_relevance, ks=(5, 10, 100))
            for key in witness_predictions:
                witness_predictions[key][test] = witness_metrics[key]
                random_predictions[key][test] = random_metrics[key]
            selected_tokens = sum(len(value) for value in plan)
            fold_rows.append(
                {
                    "fold": fold,
                    "fit_queries": len(fit),
                    "test_queries": len(test),
                    "selected_tokens": selected_tokens,
                    "token_fraction": selected_tokens / int(document_counts.sum()),
                    "min_tokens_per_document": min(len(value) for value in plan),
                    "median_tokens_per_document": float(np.median([len(value) for value in plan])),
                    "max_tokens_per_document": max(len(value) for value in plan),
                    "witness_score_ms": witness_ms,
                    "random_score_ms": random_ms,
                }
            )
        reports[str(threshold)] = {
            "mean_token_fraction": float(np.mean([row["token_fraction"] for row in fold_rows])),
            "witness": {key: float(values.mean()) for key, values in witness_predictions.items()},
            "matched_random": {key: float(values.mean()) for key, values in random_predictions.items()},
            "per_query_witness": {key: values.tolist() for key, values in witness_predictions.items()},
            "folds": fold_rows,
        }
    result = {
        "protocol": "query-hash five-fold token-witness crossfit",
        "compiler_uses_qrels": False,
        "winner_tensor_ms": winner_ms,
        "queries": len(queries),
        "documents": len(documents),
        "full_tokens": int(document_counts.sum()),
        "minimum_tokens": args.minimum_tokens,
        "full_trace": {
            key: float(values.mean())
            for key, values in full_metrics.items()
            if key in {"ndcg_at_5", "ndcg_at_10", "recall_at_100"}
        },
        "thresholds": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"winner_tensor_ms": winner_ms, "thresholds": {key: {"mean_token_fraction": value["mean_token_fraction"], "witness": value["witness"], "matched_random": value["matched_random"]} for key, value in reports.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
