#!/usr/bin/env python3
"""Audit Omni cascade failure boundaries without designing a new policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

from tools.analyze_omni_adaptive_cascade import load_scored_ranking
from tools.analyze_omni_pair import _query_metrics, load_qrels, load_ranking


DEPTHS = (20, 50, 100)
FULL_FALLBACK_ROWS = 1110


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in queries:
                raise ValueError(f"duplicate query at {path}:{line_number}")
            queries[query_id] = str(row["query"])
    return queries


def minimum_containment_depth(
    teacher_topk: Iterable[str],
    locator_ranking: list[str],
    *,
    full_fallback_rows: int = FULL_FALLBACK_ROWS,
) -> int:
    """Return the first candidate depth containing every teacher item."""
    teacher = set(teacher_topk)
    for depth in DEPTHS:
        if teacher <= set(locator_ranking[:depth]):
            return depth
    return full_fallback_rows


def candidate_recall(candidates: Iterable[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("candidate recall requires at least one relevant document")
    return len(set(candidates) & relevant_set) / len(relevant_set)


def area_under_roc(values: Iterable[float], targets: Iterable[bool]) -> float:
    """Compute tie-aware AUROC; 0.5 means the marker has no ordering power."""
    scores = np.asarray(list(values), dtype=np.float64)
    labels = np.asarray(list(targets), dtype=bool)
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        raise ValueError("AUROC requires both positive and negative examples")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def minimum_quality_depth(
    row: dict[str, Any],
    tolerance: float = 0.001,
    *,
    full_fallback_rows: int = FULL_FALLBACK_ROWS,
) -> int:
    for depth in DEPTHS:
        if row["regret"][f"cascade{depth}_ndcg_at_10"] <= tolerance:
            return depth
    return full_fallback_rows


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float(np.mean(rows)) if rows else 0.0


def _percentile(values: Iterable[float], level: float) -> float:
    rows = list(values)
    return float(np.percentile(rows, level)) if rows else 0.0


def _median(values: Iterable[float]) -> float:
    rows = list(values)
    return float(median(rows)) if rows else 0.0


def _metric(ranking: list[str], relevance: dict[str, float], name: str) -> float:
    return _query_metrics(ranking, relevance, (10, 100))[name]


def _profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "queries": len(rows),
        "query_words": {
            "mean": _mean(row["query_words"] for row in rows),
            "median": _median(row["query_words"] for row in rows),
            "p90": _percentile((row["query_words"] for row in rows), 90),
        },
        "positive_qrels": {
            "mean": _mean(row["positive_qrels"] for row in rows),
            "median": _median(row["positive_qrels"] for row in rows),
            "p90": _percentile((row["positive_qrels"] for row in rows), 90),
        },
        "mean_ndcg_at_10": {
            name: _mean(row["ndcg_at_10"][name] for row in rows)
            for name in ("full", "hpool", "agc", "cascade20", "cascade50", "cascade100")
        },
        "mean_full_minus_hpool_ndcg_at_10": _mean(
            row["regret"]["hpool_ndcg_at_10"] for row in rows
        ),
        "mean_hpool_candidate_recall_at_100": _mean(
            row["candidate_recall_at_100"]["hpool"] for row in rows
        ),
        "mean_union_candidate_recall_at_100": _mean(
            row["candidate_recall_at_100"]["hpool_agc_union"] for row in rows
        ),
        "ranking_escape_queries_at_100": sum(row["ranking_escape"] for row in rows),
        "teacher_evidence_escape_queries_at_100": sum(
            row["teacher_evidence_escape"] for row in rows
        ),
        "hpool_has_no_relevant_candidate_at_100": sum(
            row["candidate_recall_at_100"]["hpool"] == 0.0 for row in rows
        ),
    }


def analyze(
    *,
    qrels_path: Path,
    queries_path: Path,
    full_path: Path,
    hpool_path: Path,
    agc_path: Path,
    cascade_paths: dict[int, Path],
    corpus_pages: int,
    full_index_bytes: int,
    hpool_index_bytes: int,
    agc_index_bytes: int,
) -> dict[str, Any]:
    qrels = load_qrels(qrels_path)
    queries = load_queries(queries_path)
    full = load_ranking(full_path, expected_depth=100)
    hpool = load_ranking(hpool_path, expected_depth=100)
    hpool_scored = load_scored_ranking(hpool_path, expected_depth=100)
    agc = load_ranking(agc_path, expected_depth=100)
    cascades = {
        depth: load_ranking(path, expected_depth=depth)
        for depth, path in cascade_paths.items()
    }
    expected = set(qrels)
    inputs = {"queries": set(queries), "full": set(full), "hpool": set(hpool), "agc": set(agc)}
    inputs.update({f"cascade{depth}": set(rows) for depth, rows in cascades.items()})
    if any(query_ids != expected for query_ids in inputs.values()):
        raise ValueError("all input query ID sets must match qrels exactly")
    if set(cascades) != set(DEPTHS):
        raise ValueError(f"cascade paths must cover {DEPTHS}")

    query_ids = sorted(expected, key=lambda value: (not value.isdigit(), value))
    per_query: list[dict[str, Any]] = []
    escape_details: list[dict[str, Any]] = []
    for query_id in query_ids:
        relevance = qrels[query_id]
        full_docs = full[query_id]
        hpool_docs = hpool[query_id]
        hpool_scores = np.asarray([score for _, score in hpool_scored[query_id]])
        hpool_scale = float(hpool_scores.std()) + 1e-9
        agc_docs = agc[query_id]
        full_top10 = full_docs[:10]
        hpool100 = set(hpool_docs)
        agc100 = set(agc_docs)
        missing = [doc_id for doc_id in full_top10 if doc_id not in hpool100]
        evidence_missing = [doc_id for doc_id in missing if doc_id in relevance]
        depth = minimum_containment_depth(
            full_top10,
            hpool_docs,
            full_fallback_rows=corpus_pages,
        )
        rankings = {
            "full": full_docs,
            "hpool": hpool_docs,
            "agc": agc_docs,
            **{f"cascade{level}": cascades[level][query_id] for level in DEPTHS},
        }
        ndcg = {
            name: _metric(ranking, relevance, "ndcg_at_10")
            for name, ranking in rankings.items()
        }
        row = {
            "query_id": query_id,
            "query": queries[query_id],
            "query_words": len(queries[query_id].split()),
            "query_characters": len(queries[query_id]),
            "positive_qrels": len(relevance),
            "teacher_containment_depth": depth,
            "ranking_escape": bool(missing),
            "teacher_evidence_escape": bool(evidence_missing),
            "missing_full_top10_documents": missing,
            "missing_relevant_full_top10_documents": evidence_missing,
            "ndcg_at_10": ndcg,
            "regret": {
                "hpool_ndcg_at_10": ndcg["full"] - ndcg["hpool"],
                **{
                    f"cascade{level}_ndcg_at_10": ndcg["full"] - ndcg[f"cascade{level}"]
                    for level in DEPTHS
                },
            },
            "candidate_recall_at_100": {
                "hpool": candidate_recall(hpool100, relevance),
                "agc": candidate_recall(agc100, relevance),
                "hpool_agc_union": candidate_recall(hpool100 | agc100, relevance),
            },
            "hpool_agc_overlap_at_100": len(hpool100 & agc100) / 100.0,
            "hpool_agc_union_candidates": len(hpool100 | agc100),
            "observable_markers": {
                "query_words": float(len(queries[query_id].split())),
                "hpool_standard_margin_at_10": float((hpool_scores[9] - hpool_scores[10]) / hpool_scale),
                "hpool_standard_margin_at_20": float((hpool_scores[19] - hpool_scores[20]) / hpool_scale),
                "hpool_standard_margin_at_50": float((hpool_scores[49] - hpool_scores[50]) / hpool_scale),
                "hpool_agc_disagreement_at_10": 1.0 - len(set(hpool_docs[:10]) & set(agc_docs[:10])) / 10.0,
                "hpool_agc_disagreement_at_20": 1.0 - len(set(hpool_docs[:20]) & set(agc_docs[:20])) / 20.0,
                "hpool_agc_disagreement_at_100": 1.0 - len(hpool100 & agc100) / 100.0,
            },
        }
        row["quality_preservation_depth"] = minimum_quality_depth(
            row,
            full_fallback_rows=corpus_pages,
        )
        per_query.append(row)
        if missing:
            full_positions = {doc_id: rank for rank, doc_id in enumerate(full_docs, 1)}
            hpool_positions = {doc_id: rank for rank, doc_id in enumerate(hpool_docs, 1)}
            agc_positions = {doc_id: rank for rank, doc_id in enumerate(agc_docs, 1)}
            escape_details.append(
                {
                    "query_id": query_id,
                    "query": queries[query_id],
                    "positive_qrels": sorted(relevance),
                    "positive_qrel_ranks_at_100": [
                        {
                            "doc_id": doc_id,
                            "full": full_positions.get(doc_id),
                            "hpool": hpool_positions.get(doc_id),
                            "agc": agc_positions.get(doc_id),
                        }
                        for doc_id in sorted(relevance)
                    ],
                    "full_ndcg_at_10": ndcg["full"],
                    "hpool_ndcg_at_10": ndcg["hpool"],
                    "cascade100_ndcg_at_10": ndcg["cascade100"],
                    "missing_documents": [
                        {
                            "doc_id": doc_id,
                            "full_rank": full_positions[doc_id],
                            "is_qrel_relevant": doc_id in relevance,
                            "agc_rank_at_100": agc_positions.get(doc_id),
                        }
                        for doc_id in missing
                    ],
                    "hpool_relevant_candidate_recall_at_100": row["candidate_recall_at_100"]["hpool"],
                    "union_relevant_candidate_recall_at_100": row["candidate_recall_at_100"]["hpool_agc_union"],
                }
            )

    groups = {
        str(depth): [row for row in per_query if row["teacher_containment_depth"] == depth]
        for depth in (*DEPTHS, corpus_pages)
    }
    missing_docs = [item for row in escape_details for item in row["missing_documents"]]
    hpool_recalls = [row["candidate_recall_at_100"]["hpool"] for row in per_query]
    union_recalls = [row["candidate_recall_at_100"]["hpool_agc_union"] for row in per_query]
    union_sizes = [row["hpool_agc_union_candidates"] for row in per_query]
    combined_locator_bytes = hpool_index_bytes + agc_index_bytes
    marker_names = list(per_query[0]["observable_markers"])
    signal_targets = {
        "requires_more_than_20_for_teacher_containment": [
            row["teacher_containment_depth"] > 20 for row in per_query
        ],
        "requires_more_than_50_for_teacher_containment": [
            row["teacher_containment_depth"] > 50 for row in per_query
        ],
        "teacher_ranking_escape_at_100": [row["ranking_escape"] for row in per_query],
    }
    signals: dict[str, Any] = {}
    for target_name, targets in signal_targets.items():
        positives = sum(targets)
        if positives in (0, len(targets)):
            signals[target_name] = {
                "positives": positives,
                "markers": [],
                "unavailable_reason": "AUROC requires both positive and negative examples",
            }
            continue
        signal_rows = []
        for marker_name in marker_names:
            auc = area_under_roc(
                (row["observable_markers"][marker_name] for row in per_query), targets
            )
            signal_rows.append(
                {
                    "marker": marker_name,
                    "auc": auc,
                    "oriented_auc": max(auc, 1.0 - auc),
                    "risk_direction": "higher" if auc >= 0.5 else "lower",
                    "requires_agc": "hpool_agc" in marker_name,
                }
            )
        signals[target_name] = {
            "positives": positives,
            "markers": sorted(signal_rows, key=lambda row: row["oriented_auc"], reverse=True),
        }
    containment_quality_cross_tab = Counter(
        (row["teacher_containment_depth"], row["quality_preservation_depth"])
        for row in per_query
    )
    hpool_harmed = [
        row for row in per_query if row["regret"]["hpool_ndcg_at_10"] > 0.001
    ]
    return {
        "analysis_scope": {
            "purpose": "post_hoc_failure_boundary_audit_not_a_deployable_policy",
            "queries": len(query_ids),
            "corpus_pages": corpus_pages,
            "teacher_target": "contain_Full_top10_in_locator_candidates",
            "qrels_use": "analysis_only",
        },
        "input_sha256": {
            "qrels": _sha256(qrels_path),
            "queries": _sha256(queries_path),
            "full": _sha256(full_path),
            "hpool": _sha256(hpool_path),
            "agc": _sha256(agc_path),
            **{f"cascade{depth}": _sha256(path) for depth, path in cascade_paths.items()},
        },
        "teacher_containment_groups": {
            "counts": dict(sorted(Counter(row["teacher_containment_depth"] for row in per_query).items())),
            "profiles": {depth: _profile(rows) for depth, rows in groups.items()},
        },
        "quality_preservation_groups": {
            "definition": "first cascade depth with Full_minus_cascade_nDCG@10 <= 0.001",
            "counts": dict(sorted(Counter(row["quality_preservation_depth"] for row in per_query).items())),
            "teacher_containment_by_quality_depth": [
                {
                    "teacher_containment_depth": containment_depth,
                    "quality_preservation_depth": quality_depth,
                    "queries": count,
                }
                for (containment_depth, quality_depth), count in sorted(containment_quality_cross_tab.items())
            ],
        },
        "failure_mechanism_summary": {
            "ndcg_harm_tolerance": 0.001,
            "hpool_quality_harm_queries": len(hpool_harmed),
            "hpool_quality_harm_with_full_top10_contained_at_100": sum(
                not row["ranking_escape"] for row in hpool_harmed
            ),
            "hpool_quality_harm_recovered_by_cascade100": sum(
                row["regret"]["cascade100_ndcg_at_10"] <= 0.001
                for row in hpool_harmed
            ),
            "cascade_quality_harm_queries": {
                str(depth): sum(
                    row["regret"][f"cascade{depth}_ndcg_at_10"] > 0.001
                    for row in per_query
                )
                for depth in DEPTHS
            },
            "irreducible_hpool_candidate_failure_query_ids": [
                row["query_id"]
                for row in per_query
                if row["regret"]["cascade100_ndcg_at_10"] > 0.001
            ],
        },
        "escape_audit": {
            "ranking_escape_queries": len(escape_details),
            "ranking_escape_documents": len(missing_docs),
            "ranking_escape_documents_recovered_by_agc_at_100": sum(
                item["agc_rank_at_100"] is not None for item in missing_docs
            ),
            "teacher_evidence_escape_queries": sum(row["teacher_evidence_escape"] for row in per_query),
            "teacher_evidence_escape_documents": sum(item["is_qrel_relevant"] for item in missing_docs),
            "teacher_evidence_escape_documents_recovered_by_agc_at_100": sum(
                item["is_qrel_relevant"] and item["agc_rank_at_100"] is not None
                for item in missing_docs
            ),
            "details": escape_details,
        },
        "relevant_candidate_boundary": {
            "hpool_mean_recall_at_100": _mean(hpool_recalls),
            "agc_mean_recall_at_100": _mean(
                row["candidate_recall_at_100"]["agc"] for row in per_query
            ),
            "union_mean_recall_at_100": _mean(union_recalls),
            "queries_improved_by_union": sum(union > base for base, union in zip(hpool_recalls, union_recalls, strict=True)),
            "queries_with_zero_hpool_relevant_candidates": sum(value == 0.0 for value in hpool_recalls),
            "queries_with_zero_union_relevant_candidates": sum(value == 0.0 for value in union_recalls),
            "queries_with_complete_hpool_relevant_coverage": sum(value == 1.0 for value in hpool_recalls),
            "queries_with_complete_union_relevant_coverage": sum(value == 1.0 for value in union_recalls),
        },
        "multi_locator_cost_ledger": {
            "hpool_index_bytes": hpool_index_bytes,
            "agc_index_bytes": agc_index_bytes,
            "combined_locator_index_bytes": combined_locator_bytes,
            "combined_over_hpool_ratio": combined_locator_bytes / hpool_index_bytes,
            "full_over_combined_ratio": full_index_bytes / combined_locator_bytes,
            "union_candidates_per_query": {
                "mean": _mean(union_sizes),
                "median": float(median(union_sizes)),
                "p95": _percentile(union_sizes, 95),
                "maximum": max(union_sizes),
            },
            "extra_candidates_over_hpool100_mean": _mean(value - 100 for value in union_sizes),
            "not_measured": [
                "second_checkpoint_weight_residency",
                "second_model_lifecycle_and_versioning",
                "second_query_encoding_latency",
                "fusion_and_deduplication_latency",
            ],
            "interpretation": "candidate recall gains are diagnostic only until the unmeasured costs are charged",
        },
        "observable_risk_signal_audit": {
            "method": "post_hoc_univariate_AUROC_no_feature_selection_claim",
            "warning": "AGC disagreement markers require a second model and are not free deployment signals",
            "targets": signals,
        },
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--hpool", type=Path, required=True)
    parser.add_argument("--agc", type=Path, required=True)
    for depth in DEPTHS:
        parser.add_argument(f"--cascade{depth}", type=Path, required=True)
    parser.add_argument("--corpus-pages", type=int, required=True)
    parser.add_argument("--full-index-bytes", type=int, required=True)
    parser.add_argument("--hpool-index-bytes", type=int, required=True)
    parser.add_argument("--agc-index-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        qrels_path=args.qrels,
        queries_path=args.queries,
        full_path=args.full,
        hpool_path=args.hpool,
        agc_path=args.agc,
        cascade_paths={depth: getattr(args, f"cascade{depth}") for depth in DEPTHS},
        corpus_pages=args.corpus_pages,
        full_index_bytes=args.full_index_bytes,
        hpool_index_bytes=args.hpool_index_bytes,
        agc_index_bytes=args.agc_index_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "groups": report["teacher_containment_groups"]["counts"],
        "escape_audit": {key: value for key, value in report["escape_audit"].items() if key != "details"},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
