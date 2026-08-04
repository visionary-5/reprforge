#!/usr/bin/env python3
"""Cross-fit pooled MaxSim with a workload-prototype residual score sketch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analyze_residual_token_witness import _residual_tensor
from analyze_token_witness_index import _embedding_views, _fold
from reprforge.heterogeneity_atlas import query_metrics
from reprforge.heterogeneous_index import _load_shard
from reprforge.workload_probe_compiler import fit_spherical_probes


def _corrected_score_surface(
    queries,
    documents,
    probes: np.ndarray,
    corrections: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    import torch

    torch_device = torch.device(device)
    assignments = tuple(
        np.argmax(np.asarray(query, dtype=np.float32) @ probes.T, axis=1).astype(
            np.int64
        )
        for query in queries
    )
    scores = np.zeros((len(queries), len(documents)), dtype=np.float32)
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
        correction_batch = torch.tensor(
            corrections[document_start : document_start + len(document_values)],
            dtype=torch.float32,
            device=torch_device,
        )
        for query_start in range(0, len(queries), batch_size):
            query_values = queries[query_start : query_start + batch_size]
            assignment_values = assignments[query_start : query_start + batch_size]
            query_lengths = torch.tensor(
                [len(value) for value in query_values], device=torch_device
            )
            padded_queries = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(np.asarray(value), dtype=torch.float32) for value in query_values],
                batch_first=True,
            ).to(torch_device)
            padded_assignments = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(value, dtype=torch.int64) for value in assignment_values],
                batch_first=True,
            ).to(torch_device)
            query_positions = torch.arange(
                padded_queries.shape[1], device=torch_device
            )
            with torch.inference_mode():
                similarities = torch.einsum(
                    "aqd,bkd->abqk", padded_queries, padded_documents
                ).masked_fill(
                    document_positions[None, None, None, :]
                    >= document_lengths[None, :, None, None],
                    float("-inf"),
                )
                maxsim = similarities.max(dim=-1).values
                selected_corrections = correction_batch[:, padded_assignments].permute(
                    1, 0, 2
                )
                query_mask = query_positions[None, :] < query_lengths[:, None]
                values = (
                    (maxsim + selected_corrections) * query_mask[:, None, :]
                ).sum(dim=-1)
            scores[
                query_start : query_start + len(query_values),
                document_start : document_start + len(document_values),
            ] = values.cpu().numpy()
    torch.cuda.synchronize(torch_device)
    return scores, (time.perf_counter() - began) * 1000.0


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
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--fit-scale", action="store_true")
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
    queries = _embedding_views(query_vectors, query_offsets)
    full_documents = _embedding_views(full_vectors, full_offsets)
    pooled_documents = _embedding_views(pooled_vectors, pooled_offsets)
    relevance = [dict() for _ in query_ids]
    for query, corpus, value in zip(
        labels["query_positions"],
        labels["corpus_positions"],
        labels["relevance"],
        strict=True,
    ):
        relevance[int(query)][int(corpus)] = float(value)
    folds = np.asarray([_fold(value) for value in query_ids], dtype=np.int16)
    predictions = {
        "ndcg_at_5": np.zeros(len(queries)),
        "ndcg_at_10": np.zeros(len(queries)),
        "recall_at_100": np.zeros(len(queries)),
    }
    shuffled_predictions = {key: np.zeros(len(queries)) for key in predictions}
    fold_rows = []
    for fold in sorted(set(folds.tolist())):
        fit = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        fit_tokens = np.concatenate([queries[index] for index in fit], axis=0)
        probes, quantization_errors = fit_spherical_probes(
            fit_tokens,
            count=args.probe_count,
            seed=20260803 + fold + args.probe_count * 100,
            iterations=args.kmeans_iterations,
        )
        _, residuals, residual_ms = _residual_tensor(
            (probes,),
            full_documents,
            pooled_documents,
            device=args.device,
            batch_size=args.batch_size,
        )
        corrections = residuals[0].astype(np.float32)
        generator = np.random.default_rng(20260803 + fold + args.probe_count * 1000)
        shuffled = corrections[generator.permutation(len(corrections))]
        scale = 1.0
        calibration_ms = 0.0
        if args.fit_scale:
            fit_scores, calibration_ms = _corrected_score_surface(
                tuple(queries[index] for index in fit),
                pooled_documents,
                probes,
                corrections,
                device=args.device,
                batch_size=args.batch_size,
            )
            pool_fit = np.asarray(pooled_runtime["scores"])[fit]
            target_delta = np.asarray(full_runtime["scores"])[fit] - pool_fit
            predicted_delta = fit_scores - pool_fit
            denominator = float(np.square(predicted_delta).sum())
            if denominator > 0:
                scale = max(
                    0.0,
                    float((predicted_delta * target_delta).sum()) / denominator,
                )
        test_queries = tuple(queries[index] for index in test)
        scores, score_ms = _corrected_score_surface(
            test_queries,
            pooled_documents,
            probes,
            corrections * scale,
            device=args.device,
            batch_size=args.batch_size,
        )
        shuffled_scores, shuffled_ms = _corrected_score_surface(
            test_queries,
            pooled_documents,
            probes,
            shuffled * scale,
            device=args.device,
            batch_size=args.batch_size,
        )
        test_relevance = tuple(relevance[index] for index in test)
        metrics = query_metrics(scores, test_relevance, ks=(5, 10, 100))
        shuffled_metrics = query_metrics(
            shuffled_scores, test_relevance, ks=(5, 10, 100)
        )
        for metric in predictions:
            predictions[metric][test] = metrics[metric]
            shuffled_predictions[metric][test] = shuffled_metrics[metric]
        fold_rows.append(
            {
                "fold": fold,
                "probe_quantization_mean": float(quantization_errors.mean()),
                "probe_quantization_p95": float(
                    np.quantile(quantization_errors, 0.95)
                ),
                "correction_mean": float(corrections.mean()),
                "correction_p05": float(np.quantile(corrections, 0.05)),
                "correction_p95": float(np.quantile(corrections, 0.95)),
                "fit_scale": scale,
                "calibration_score_ms": calibration_ms,
                "residual_tensor_ms": residual_ms,
                "score_ms": score_ms,
                "shuffled_score_ms": shuffled_ms,
            }
        )

    baselines = {}
    for name, runtime in (("full", full_runtime), ("pool9", pooled_runtime)):
        values = query_metrics(runtime["scores"], tuple(relevance), ks=(5, 10, 100))
        baselines[name] = {
            key: float(metric.mean())
            for key, metric in values.items()
            if key in predictions
        }
    full_vector_bytes = int(full_vectors.size * full_vectors.dtype.itemsize)
    pool_vector_bytes = int(pooled_vectors.size * pooled_vectors.dtype.itemsize)
    correction_bytes = len(item_ids) * args.probe_count * 2
    result = {
        "protocol": (
            "qrel-free spherical workload probes; pool9 MaxSim plus nearest-"
            "probe full-minus-pool scalar correction; optional fit-workload "
            "teacher scale; query-hash five-fold crossfit"
        ),
        "compiler_uses_qrels": False,
        "fit_scale": args.fit_scale,
        "probe_count": args.probe_count,
        "storage": {
            "full_vector_bytes": full_vector_bytes,
            "pool_vector_bytes": pool_vector_bytes,
            "correction_float16_bytes": correction_bytes,
            "compiled_fraction_of_full": (
                pool_vector_bytes + correction_bytes
            )
            / full_vector_bytes,
        },
        "baselines": baselines,
        "score_sketch": {
            name: float(values.mean()) for name, values in predictions.items()
        },
        "shuffled_document_control": {
            name: float(values.mean())
            for name, values in shuffled_predictions.items()
        },
        "per_query_score_sketch": {
            name: values.tolist() for name, values in predictions.items()
        },
        "folds": fold_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "storage": result["storage"],
                "baselines": baselines,
                "score_sketch": result["score_sketch"],
                "shuffled_document_control": result["shuffled_document_control"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
