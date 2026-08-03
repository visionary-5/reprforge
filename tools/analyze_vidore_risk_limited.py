#!/usr/bin/env python3
"""Transfer frozen risk-limited acquisition to ViDoRe score traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.risk_limited_acquisition import (
    build_candidate_surface,
    crossfit_boundary_acquisition,
    crossfit_candidate_set_acquisition,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_runtime(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "query_ids",
            "corpus_ids",
            "scores",
            "vector_bytes",
            "vector_counts",
            "query_vector_counts",
            "encode_ms",
            "index_total_ms",
        }
        if not required <= set(payload.files):
            raise ValueError(f"{path} lacks runtime trace fields")
        return {key: np.asarray(payload[key]) for key in payload.files}


def _qrels(
    path: Path,
    *,
    query_count: int,
) -> list[dict[int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        query = np.asarray(payload["query_positions"], dtype=np.int32)
        corpus = np.asarray(payload["corpus_positions"], dtype=np.int32)
        relevance = np.asarray(payload["relevance"], dtype=np.int16)
    output: list[dict[int, int]] = [dict() for _ in range(query_count)]
    for q, d, value in zip(query, corpus, relevance, strict=True):
        output[int(q)][int(d)] = int(value)
    if any(not values for values in output):
        raise ValueError("every query must contain at least one qrel")
    return output


def _complete_rankings(
    topk: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(candidate_indices)
    for query, selected in enumerate(topk):
        chosen = set(int(value) for value in selected)
        tail = [
            int(value)
            for value in candidate_indices[query]
            if int(value) not in chosen
        ]
        output[query] = np.asarray([*selected.tolist(), *tail], dtype=np.int32)
    return output


def _metrics(
    rankings: np.ndarray,
    qrels: Sequence[dict[int, int]],
) -> dict[str, float]:
    ndcg: list[float] = []
    recall: list[float] = []
    for ranking, relevance in zip(rankings, qrels, strict=True):
        dcg = sum(
            (2 ** relevance.get(int(page), 0) - 1) / np.log2(rank + 2)
            for rank, page in enumerate(ranking[:10])
        )
        ideal = sorted(relevance.values(), reverse=True)[:10]
        idcg = sum(
            (2**value - 1) / np.log2(rank + 2)
            for rank, value in enumerate(ideal)
        )
        ndcg.append(float(dcg / idcg) if idcg else 0.0)
        relevant = set(relevance)
        recall.append(
            len(relevant & {int(page) for page in ranking[:100]}) / len(relevant)
        )
    return {"ndcg@10": float(np.mean(ndcg)), "recall@100": float(np.mean(recall))}


def _work(
    acquired_pages: Sequence[Sequence[int]],
    encode_ms: np.ndarray,
    vector_bytes: np.ndarray,
) -> dict[str, float | int]:
    unique = {int(page) for query in acquired_pages for page in query}
    return {
        "candidate_events": sum(len(query) for query in acquired_pages),
        "mean_acquired_pages": float(np.mean([len(query) for query in acquired_pages])),
        "unique_pages": len(unique),
        "unique_visual_build_ms": float(sum(float(encode_ms[page]) for page in unique)),
        "unique_visual_bytes": int(sum(int(vector_bytes[page]) for page in unique)),
    }


def _fixed_rankings(
    candidates: np.ndarray,
    teacher_scores: np.ndarray,
    corpus_ids: Sequence[str],
    *,
    fixed_k: int,
    cutoff: int,
) -> np.ndarray:
    output = np.empty((len(candidates), cutoff), dtype=np.int32)
    for query in range(len(candidates)):
        pages = candidates[query, :fixed_k]
        order = sorted(
            range(fixed_k),
            key=lambda offset: (
                -float(teacher_scores[query, offset]),
                corpus_ids[int(pages[offset])],
            ),
        )[:cutoff]
        output[query] = pages[np.asarray(order, dtype=np.int32)]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locator-runtime", type=Path, required=True)
    parser.add_argument("--visual-runtime", type=Path, required=True)
    parser.add_argument("--oracle-labels", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    locator = _load_runtime(args.locator_runtime)
    visual = _load_runtime(args.visual_runtime)
    query_ids = [str(value) for value in locator["query_ids"]]
    corpus_ids = [str(value) for value in locator["corpus_ids"]]
    if query_ids != [str(value) for value in visual["query_ids"]]:
        raise ValueError("locator and visual query orders differ")
    if corpus_ids != [str(value) for value in visual["corpus_ids"]]:
        raise ValueError("locator and visual corpus orders differ")
    labels = _qrels(args.oracle_labels, query_count=len(query_ids))
    surface = build_candidate_surface(
        corpus_ids,
        locator["scores"],
        visual["scores"],
        query_token_counts=np.asarray(
            visual["query_vector_counts"], dtype=np.int32
        ),
        page_text_token_counts=np.asarray(locator["vector_counts"], dtype=np.int32),
        candidate_pool=args.candidate_pool,
    )
    # ViDoRe does not expose source-document groups for queries.  Query IDs are
    # assigned deterministically to balanced folds; this limitation is explicit.
    score = crossfit_boundary_acquisition(
        surface,
        corpus_ids,
        query_ids,
        cutoff=args.cutoff,
        alpha=args.alpha,
    )
    candidate_set = crossfit_candidate_set_acquisition(
        surface,
        corpus_ids,
        query_ids,
        cutoff=args.cutoff,
        alpha=args.alpha,
    )
    encode_ms = np.asarray(visual["encode_ms"], dtype=np.float64)
    vector_bytes = np.asarray(visual["vector_bytes"], dtype=np.int64)

    runs: dict[str, Any] = {}
    for name, result, coverage_name, coverage in (
        (
            "risk_limited_score_envelope",
            score,
            "simultaneous_upper_coverage",
            score.coverage,
        ),
        (
            "risk_limited_candidate_set",
            candidate_set,
            "topk_set_coverage",
            candidate_set.topk_covered,
        ),
    ):
        complete = _complete_rankings(result.rankings, surface.candidate_indices)
        runs[name] = {
            "quality": _metrics(complete, labels),
            "teacher_exact_topk_set_agreement": float(
                np.mean(
                    [
                        set(row) == set(reference)
                        for row, reference in zip(
                            result.rankings,
                            result.teacher_rankings,
                            strict=True,
                        )
                    ]
                )
            ),
            "risk": {
                "alpha": args.alpha,
                coverage_name: float(coverage.mean()),
            },
            "work": _work(result.acquired_pages, encode_ms, vector_bytes),
            "folds": list(result.folds),
        }

    for fixed_k in (10, 20, 50, 100):
        if fixed_k > args.candidate_pool or fixed_k < args.cutoff:
            continue
        topk = _fixed_rankings(
            surface.candidate_indices,
            score.teacher_scores,
            corpus_ids,
            fixed_k=fixed_k,
            cutoff=args.cutoff,
        )
        acquired = surface.candidate_indices[:, :fixed_k]
        runs[f"fixed_k{fixed_k}"] = {
            "quality": _metrics(_complete_rankings(topk, surface.candidate_indices), labels),
            "teacher_exact_topk_set_agreement": float(
                np.mean(
                    [
                        set(row) == set(reference)
                        for row, reference in zip(topk, score.teacher_rankings, strict=True)
                    ]
                )
            ),
            "work": _work(acquired, encode_ms, vector_bytes),
        }

    teacher_complete = _complete_rankings(score.teacher_rankings, surface.candidate_indices)
    runs["full_candidate_teacher"] = {
        "quality": _metrics(teacher_complete, labels),
        "teacher_exact_topk_set_agreement": 1.0,
        "work": _work(surface.candidate_indices, encode_ms, vector_bytes),
    }
    proposed = runs["risk_limited_candidate_set"]
    eligible = [
        (name, record)
        for name, record in runs.items()
        if name.startswith("fixed_k")
        and record["quality"]["ndcg@10"]
        >= proposed["quality"]["ndcg@10"] - 0.005
    ]
    best_name, best = min(
        eligible,
        key=lambda pair: pair[1]["work"]["unique_visual_build_ms"],
    )
    build_reduction = 1.0 - proposed["work"]["unique_visual_build_ms"] / best[
        "work"
    ]["unique_visual_build_ms"]
    teacher_quality = runs["full_candidate_teacher"]["quality"]
    gate = {
        "topk_coverage_at_least_0_93": proposed["risk"]["topk_set_coverage"] >= 0.93,
        "ndcg_within_0_005_of_teacher": proposed["quality"]["ndcg@10"]
        >= teacher_quality["ndcg@10"] - 0.005,
        "build_time_reduction_at_least_0_20": build_reduction >= 0.20,
    }
    output = {
        "schema_version": 1,
        "dataset": args.dataset,
        "queries": len(query_ids),
        "corpus": len(corpus_ids),
        "candidate_pool": args.candidate_pool,
        "cutoff": args.cutoff,
        "locator": "full-corpus cheap text late-interaction trace",
        "policy_uses_qrels": False,
        "runs": runs,
        "comparison": {
            "quality_matched_best_fixed": best_name,
            "unique_visual_build_time_reduction": build_reduction,
        },
        "gate": {**gate, "passed": all(gate.values())},
        "artifacts": {
            "locator_runtime_sha256": _sha256(args.locator_runtime),
            "visual_runtime_sha256": _sha256(args.visual_runtime),
            "oracle_labels_sha256": _sha256(args.oracle_labels),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
