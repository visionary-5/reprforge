#!/usr/bin/env python3
"""Evaluate progressive visual-evidence acquisition on IRPAPERS scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.progressive_evidence import (
    apply_progressive_policy,
    build_evidence_stages,
    paper_disjoint_bm25_margin_router,
    paper_disjoint_progressive_probe,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality(
    rankings: np.ndarray,
    corpus_ids: Sequence[str],
    gold_ids: Sequence[str],
) -> dict[str, float]:
    return {
        f"recall_{cutoff}": float(
            np.mean(
                [
                    gold in {corpus_ids[int(value)] for value in row[:cutoff]}
                    for row, gold in zip(rankings, gold_ids, strict=True)
                ]
            )
        )
        for cutoff in (1, 5, 20)
    }


def _cost(
    selected_depths: np.ndarray,
    base_order: np.ndarray,
) -> dict[str, Any]:
    touched: set[int] = set()
    for query, depth in enumerate(selected_depths):
        touched.update(int(value) for value in base_order[query, : int(depth)])
    values, counts = np.unique(selected_depths, return_counts=True)
    return {
        "candidate_events": int(selected_depths.sum()),
        "mean_acquired_pages": float(selected_depths.mean()),
        "unique_candidate_pages": len(touched),
        "selection_counts": {
            str(int(value)): int(count)
            for value, count in zip(values, counts, strict=True)
        },
    }


def _agreement(rankings: np.ndarray, teacher: np.ndarray) -> dict[str, float | int]:
    top1_disagreement = rankings[:, 0] != teacher[:, 0]
    top5_exact = np.asarray(
        [
            set(row[:5]) == set(reference[:5])
            for row, reference in zip(rankings, teacher, strict=True)
        ]
    )
    return {
        "top1_disagreements": int(top1_disagreement.sum()),
        "top1_disagreement_rate": float(top1_disagreement.mean()),
        "exact_top5_set_agreement": float(top5_exact.mean()),
    }


def _run_record(
    rankings: np.ndarray,
    selected: np.ndarray,
    *,
    teacher: np.ndarray,
    base_order: np.ndarray,
    corpus_ids: Sequence[str],
    gold_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "quality": _quality(rankings, corpus_ids, gold_ids),
        "cost": _cost(selected, base_order),
        "teacher_agreement": _agreement(rankings, teacher),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-surface", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    surface = np.load(args.score_surface)
    corpus_ids = [str(value) for value in surface["corpus_ids"]]
    locator = np.asarray(surface["bm25_scores"], dtype=np.float64)
    visual = np.asarray(surface["visual_scores"], dtype=np.float64)
    with args.queries.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != locator.shape[0]:
        raise ValueError("query metadata and score surface differ in length")
    gold_ids = [str(row["dataset_id"]) for row in rows]
    source_papers = [str(row["pdf_id"]) for row in rows]

    evidence = build_evidence_stages(
        corpus_ids,
        locator,
        visual,
        top_k=20,
    )
    teacher = evidence.orders[evidence.candidate_limit]
    progressive = paper_disjoint_progressive_probe(evidence, source_papers)
    margin_router = paper_disjoint_bm25_margin_router(
        evidence,
        source_papers,
        locator,
    )
    naive_selected, naive_rankings = apply_progressive_policy(
        evidence,
        np.arange(len(rows)),
        {4: 0.0, 6: 0.0, 8: 0.0},
    )

    runs: dict[str, Any] = {}
    zero = np.zeros(len(rows), dtype=np.int16)
    runs["bm25_only"] = _run_record(
        evidence.base_order,
        zero,
        teacher=teacher,
        base_order=evidence.base_order,
        corpus_ids=corpus_ids,
        gold_ids=gold_ids,
    )
    for stage in evidence.stages[1:]:
        selected = np.full(len(rows), stage, dtype=np.int16)
        runs[f"fixed_k{stage}"] = _run_record(
            evidence.orders[stage],
            selected,
            teacher=teacher,
            base_order=evidence.base_order,
            corpus_ids=corpus_ids,
            gold_ids=gold_ids,
        )
    runs["naive_stability"] = _run_record(
        naive_rankings,
        naive_selected,
        teacher=teacher,
        base_order=evidence.base_order,
        corpus_ids=corpus_ids,
        gold_ids=gold_ids,
    )
    runs["paper_disjoint_bm25_margin_router"] = _run_record(
        margin_router["rankings"],
        margin_router["selected_depths"],
        teacher=teacher,
        base_order=evidence.base_order,
        corpus_ids=corpus_ids,
        gold_ids=gold_ids,
    )
    runs["paper_disjoint_progressive_stability_margin"] = _run_record(
        progressive["rankings"],
        progressive["selected_depths"],
        teacher=teacher,
        base_order=evidence.base_order,
        corpus_ids=corpus_ids,
        gold_ids=gold_ids,
    )

    earliest = []
    for query in range(len(rows)):
        earliest.append(
            next(
                stage
                for stage in evidence.stages
                if evidence.orders[stage][query, 0] == teacher[query, 0]
            )
        )
    earliest_array = np.asarray(earliest, dtype=np.int16)
    oracle_rankings = np.stack(
        [
            evidence.orders[int(stage)][query]
            for query, stage in enumerate(earliest_array)
        ]
    )
    runs["teacher_visible_earliest_stage_oracle"] = _run_record(
        oracle_rankings,
        earliest_array,
        teacher=teacher,
        base_order=evidence.base_order,
        corpus_ids=corpus_ids,
        gold_ids=gold_ids,
    )

    progressive_cost = runs[
        "paper_disjoint_progressive_stability_margin"
    ]["cost"]
    fixed_cost = runs["fixed_k10"]["cost"]
    output = {
        "schema_version": 1,
        "problem": (
            "Acquire expensive visual page evidence progressively and stop "
            "from observed ranking intervention, rather than choosing one K."
        ),
        "dataset": {
            "name": "IRPAPERS",
            "queries": len(rows),
            "pages": len(corpus_ids),
            "source_papers": len(set(source_papers)),
            "single_gold_page_per_query": True,
        },
        "teacher": {
            "definition": "candidate-relative BM25+ColPali fusion over BM25 Top-10",
            "uses_qrels": False,
            "is_deployable_without_historical_full_scores": False,
        },
        "policy": {
            "stages": list(evidence.stages),
            "observable_stop_signal": (
                "Top-1 identity stable across consecutive stages and fused "
                "Top-1/Top-2 margin above a training-only threshold"
            ),
            "threshold_selection": (
                "leave-one-source-paper-out; minimize training page-events "
                "subject to zero training disagreement with the Top-10 teacher"
            ),
            "uses_qrels_for_thresholds": False,
            "formal_risk_guarantee": False,
            "folds": progressive["folds"],
        },
        "runs": runs,
        "headline": {
            "candidate_event_reduction_vs_fixed_k10": 1.0
            - progressive_cost["candidate_events"] / fixed_cost["candidate_events"],
            "unique_page_reduction_vs_fixed_k10": 1.0
            - progressive_cost["unique_candidate_pages"]
            / fixed_cost["unique_candidate_pages"],
            "top1_teacher_disagreement_rate": runs[
                "paper_disjoint_progressive_stability_margin"
            ]["teacher_agreement"]["top1_disagreement_rate"],
            "recall_1_delta_vs_fixed_k10": runs[
                "paper_disjoint_progressive_stability_margin"
            ]["quality"]["recall_1"]
            - runs["fixed_k10"]["quality"]["recall_1"],
        },
        "artifact_sha256": {
            "score_surface": _sha256(args.score_surface),
            "queries": _sha256(args.queries),
        },
        "decision": (
            "MECHANISM SIGNAL, NOT YET AN ALGORITHM CLAIM: observed visual "
            "intervention supports progressive stopping better than a static "
            "BM25-margin router on this benchmark. Transfer and calibrated "
            "risk control are still required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
