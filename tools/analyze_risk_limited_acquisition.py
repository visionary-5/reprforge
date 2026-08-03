#!/usr/bin/env python3
"""Run the frozen IRPAPERS risk-limited acquisition protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.irpapers_benchmark import load_irpapers
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


def _token_count(text: str) -> int:
    return max(1, len(str(text).split()))


def _recall(
    rankings: np.ndarray,
    corpus_ids: Sequence[str],
    gold_ids: Sequence[str],
    *,
    cutoffs: Sequence[int] = (1, 5),
) -> dict[str, float]:
    return {
        f"recall_{cutoff}": float(
            np.mean(
                [
                    gold
                    in {
                        corpus_ids[int(page)]
                        for page in ranking[: min(cutoff, len(ranking))]
                    }
                    for ranking, gold in zip(rankings, gold_ids, strict=True)
                ]
            )
        )
        for cutoff in cutoffs
    }


def _agreement(rankings: np.ndarray, teacher: np.ndarray) -> dict[str, Any]:
    exact = np.asarray(
        [
            set(row) == set(reference)
            for row, reference in zip(rankings, teacher, strict=True)
        ]
    )
    return {
        "exact_topk_set_agreement": float(exact.mean()),
        "topk_set_disagreements": int((~exact).sum()),
        "top1_agreement": float(np.mean(rankings[:, 0] == teacher[:, 0])),
    }


def _fixed_k_rankings(
    candidate_indices: np.ndarray,
    teacher_scores: np.ndarray,
    corpus_ids: Sequence[str],
    *,
    fixed_k: int,
    cutoff: int,
) -> np.ndarray:
    output = np.empty((len(candidate_indices), cutoff), dtype=np.int32)
    for query in range(len(candidate_indices)):
        pages = candidate_indices[query, :fixed_k]
        scores = teacher_scores[query, :fixed_k]
        order = sorted(
            range(fixed_k),
            key=lambda offset: (
                -float(scores[offset]),
                corpus_ids[int(pages[offset])],
            ),
        )[:cutoff]
        output[query] = pages[np.asarray(order, dtype=np.int32)]
    return output


def _run_record(
    rankings: np.ndarray,
    *,
    teacher: np.ndarray,
    corpus_ids: Sequence[str],
    gold_ids: Sequence[str],
    acquired_counts: np.ndarray,
    acquired_pages: Sequence[Sequence[int]],
) -> dict[str, Any]:
    unique_pages = {int(page) for query in acquired_pages for page in query}
    return {
        "quality": _recall(rankings, corpus_ids, gold_ids),
        "teacher_agreement": _agreement(rankings, teacher),
        "work": {
            "candidate_events": int(np.sum(acquired_counts)),
            "mean_acquired_pages": float(np.mean(acquired_counts)),
            "unique_pages": len(unique_pages),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-surface", type=Path, required=True)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    frozen = np.load(args.score_surface, allow_pickle=False)
    corpus_ids = [str(value) for value in frozen["corpus_ids"]]
    query_ids = [str(value) for value in frozen["query_ids"]]
    if args.documents is not None:
        data = load_irpapers(args.documents, args.queries, decode_images=False)
        if corpus_ids != list(data.corpus_ids) or query_ids != list(data.query_ids):
            raise ValueError("IRPAPERS data order differs from the frozen score surface")
        queries = list(data.queries)
        gold_ids = [next(iter(data.qrels[query])) for query in data.query_ids]
        source_groups = [value.split("_", 1)[0] for value in gold_ids]
        page_token_counts = [_token_count(value) for value in data.corpus_texts]
        page_text_feature = "IRPAPERS transcription whitespace-token count"
    else:
        with args.queries.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"dataset_id", "pdf_id", "question"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError("query CSV lacks dataset_id, pdf_id, or question")
        if len(rows) != len(query_ids):
            raise ValueError("query CSV differs from the frozen score surface")
        queries = [str(row["question"]) for row in rows]
        gold_ids = [str(row["dataset_id"]) for row in rows]
        source_groups = [str(row["pdf_id"]) for row in rows]
        page_token_counts = [1] * len(corpus_ids)
        page_text_feature = "unavailable; constant neutral feature"
    locator = np.asarray(frozen["bm25_scores"], dtype=np.float64)
    visual = np.asarray(frozen["visual_scores"], dtype=np.float64)
    surface = build_candidate_surface(
        corpus_ids,
        locator,
        visual,
        query_token_counts=[_token_count(value) for value in queries],
        page_text_token_counts=page_token_counts,
        candidate_pool=args.candidate_pool,
    )
    score_result = crossfit_boundary_acquisition(
        surface,
        corpus_ids,
        source_groups,
        cutoff=args.cutoff,
        alpha=args.alpha,
        batch_size=args.batch_size,
    )
    set_result = crossfit_candidate_set_acquisition(
        surface,
        corpus_ids,
        source_groups,
        cutoff=args.cutoff,
        alpha=args.alpha,
    )

    runs: dict[str, Any] = {}
    score_envelope = _run_record(
        score_result.rankings,
        teacher=score_result.teacher_rankings,
        corpus_ids=corpus_ids,
        gold_ids=gold_ids,
        acquired_counts=score_result.acquired_counts,
        acquired_pages=score_result.acquired_pages,
    )
    score_envelope["risk"] = {
        "target_alpha": args.alpha,
        "query_level_simultaneous_upper_coverage": float(
            score_result.coverage.mean()
        ),
        "uncovered_queries": int((~score_result.coverage).sum()),
        "certified_queries": int(score_result.certified.sum()),
        "pool_exhausted_queries": int(score_result.exhausted.sum()),
    }
    score_envelope["folds"] = list(score_result.folds)
    runs["risk_limited_score_envelope"] = score_envelope

    proposed = _run_record(
        set_result.rankings,
        teacher=set_result.teacher_rankings,
        corpus_ids=corpus_ids,
        gold_ids=gold_ids,
        acquired_counts=set_result.acquired_counts,
        acquired_pages=set_result.acquired_pages,
    )
    proposed["risk"] = {
        "target_alpha": args.alpha,
        "query_level_topk_set_coverage": float(set_result.topk_covered.mean()),
        "uncovered_queries": int((~set_result.topk_covered).sum()),
    }
    proposed["folds"] = list(set_result.folds)
    runs["risk_limited_candidate_set"] = proposed

    for fixed_k in (5, 10, 20, 50, 100):
        if fixed_k > args.candidate_pool or fixed_k < args.cutoff:
            continue
        rankings = _fixed_k_rankings(
            surface.candidate_indices,
            score_result.teacher_scores,
            corpus_ids,
            fixed_k=fixed_k,
            cutoff=args.cutoff,
        )
        acquired = surface.candidate_indices[:, :fixed_k]
        runs[f"fixed_k{fixed_k}"] = _run_record(
            rankings,
            teacher=score_result.teacher_rankings,
            corpus_ids=corpus_ids,
            gold_ids=gold_ids,
            acquired_counts=np.full(len(query_ids), fixed_k, dtype=np.int32),
            acquired_pages=acquired,
        )

    teacher_work = np.full(len(query_ids), args.candidate_pool, dtype=np.int32)
    runs["full_candidate_teacher"] = _run_record(
        score_result.teacher_rankings,
        teacher=score_result.teacher_rankings,
        corpus_ids=corpus_ids,
        gold_ids=gold_ids,
        acquired_counts=teacher_work,
        acquired_pages=surface.candidate_indices,
    )
    quality_target = proposed["quality"]["recall_5"]
    eligible_fixed = [
        (name, record)
        for name, record in runs.items()
        if name.startswith("fixed_k")
        and record["quality"]["recall_5"] >= quality_target - 1e-12
    ]
    best_fixed_name, best_fixed = min(
        eligible_fixed,
        key=lambda pair: pair[1]["work"]["candidate_events"],
    )
    reduction = 1.0 - proposed["work"]["candidate_events"] / best_fixed["work"][
        "candidate_events"
    ]
    gate = {
        "coverage_at_least_0_93": proposed["risk"][
            "query_level_topk_set_coverage"
        ]
        >= 0.93,
        "teacher_disagreement_at_most_0_05": 1.0
        - proposed["teacher_agreement"]["exact_topk_set_agreement"]
        <= 0.05,
        "recall5_within_one_query_of_teacher": proposed["quality"]["recall_5"]
        >= runs["full_candidate_teacher"]["quality"]["recall_5"] - 1.0 / len(query_ids),
        "event_reduction_at_least_0_20": reduction >= 0.20,
    }
    output = {
        "schema_version": 1,
        "problem": (
            "Build only visual representations whose score envelopes still "
            "overlap the requested Top-k boundary."
        ),
        "dataset": {
            "name": "IRPAPERS",
            "queries": len(query_ids),
            "pages": len(corpus_ids),
            "source_papers": len(set(source_groups)),
            "candidate_pool": args.candidate_pool,
            "cutoff": args.cutoff,
        },
        "score_contract": {
            "base": "per-query BM25 z-score inside the candidate pool",
            "visual": "ColPali MaxSim divided by whitespace query-token count",
            "fusion": "base plus train-only standardized visual score",
            "qrels_used_by_policy": False,
            "page_text_feature": page_text_feature,
        },
        "runs": runs,
        "comparison": {
            "quality_matched_best_fixed": best_fixed_name,
            "candidate_event_reduction": reduction,
        },
        "gate": {**gate, "passed": all(gate.values())},
        "artifacts": {
            "score_surface_sha256": _sha256(args.score_surface),
            "documents_sha256": (
                None if args.documents is None else _sha256(args.documents)
            ),
            "queries_sha256": _sha256(args.queries),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
