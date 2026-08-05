#!/usr/bin/env python3
"""Analyze paired OmniColPress rankings with official metric semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.compression_risk_metrics import (
    bootstrap_mean_interval,
    rbo_ext,
    regret_summary,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            query_id = str(row["query_id"])
            doc_id = str(row["doc_id"])
            relevance = float(row["relevance"])
            if relevance <= 0:
                continue
            if doc_id in qrels[query_id]:
                raise ValueError(f"duplicate qrel at {path}:{line_number}")
            qrels[query_id][doc_id] = relevance
    if not qrels:
        raise ValueError(f"no positive qrels in {path}")
    return dict(qrels)


def load_ranking(path: Path, *, expected_depth: int) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = defaultdict(list)
    previous_scores: dict[str, float] = {}
    seen: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"expected 3 tab fields at {path}:{line_number}")
            query_id, doc_id, score_text = fields
            score = float(score_text)
            if not math.isfinite(score):
                raise ValueError(f"non-finite score at {path}:{line_number}")
            if doc_id in seen[query_id]:
                raise ValueError(f"duplicate ranked document at {path}:{line_number}")
            if query_id in previous_scores and score > previous_scores[query_id] + 1e-8:
                raise ValueError(f"scores increase at {path}:{line_number}")
            seen[query_id].add(doc_id)
            rankings[query_id].append(doc_id)
            previous_scores[query_id] = score
    bad_depth = {
        query_id: len(docs)
        for query_id, docs in rankings.items()
        if len(docs) != expected_depth
    }
    if bad_depth:
        raise ValueError(f"unexpected ranking depths: {bad_depth}")
    return dict(rankings)


def _query_metrics(
    ranking: list[str], relevance: dict[str, float], ks: tuple[int, ...]
) -> dict[str, float]:
    relevant = set(relevance)
    values: dict[str, float] = {}
    sorted_relevance = sorted(relevance.values(), reverse=True)
    for k in ks:
        retrieved = ranking[:k]
        values[f"recall_at_{k}"] = len(set(retrieved) & relevant) / len(relevant)
        dcg = sum(
            relevance.get(doc_id, 0.0) / math.log2(rank + 1)
            for rank, doc_id in enumerate(retrieved, start=1)
        )
        idcg = sum(
            value / math.log2(rank + 1)
            for rank, value in enumerate(sorted_relevance[:k], start=1)
        )
        values[f"ndcg_at_{k}"] = dcg / idcg if idcg else 0.0
    values["mrr_at_100"] = next(
        (
            1.0 / rank
            for rank, doc_id in enumerate(ranking[:100], start=1)
            if doc_id in relevant
        ),
        0.0,
    )
    return values


def _official_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = {
        name.lower().replace("@", "_at_"): float(value)
        for name, value in payload["evaluation_results"].items()
        if name != "num_queries"
    }
    metrics["mrr_at_100"] = metrics.pop("mrr")
    return metrics


def analyze_pair(
    qrels_path: Path,
    full_ranking_path: Path,
    compressed_ranking_path: Path,
    *,
    depth: int = 100,
    bootstrap_seed: int = 20260805,
    bootstrap_resamples: int = 10_000,
    full_results_path: Path | None = None,
    compressed_results_path: Path | None = None,
) -> dict[str, Any]:
    qrels = load_qrels(qrels_path)
    full = load_ranking(full_ranking_path, expected_depth=depth)
    compressed = load_ranking(compressed_ranking_path, expected_depth=depth)
    expected_queries = set(qrels)
    if set(full) != expected_queries or set(compressed) != expected_queries:
        raise ValueError("qrel and paired-ranking query IDs must match exactly")

    query_ids = sorted(expected_queries, key=lambda value: (not value.isdigit(), value))
    ks = tuple(k for k in (1, 5, 10, 100) if k <= depth)
    per_query: list[dict[str, Any]] = []
    metric_names: list[str] | None = None
    fidelity: dict[str, list[float]] = defaultdict(list)
    for query_id in query_ids:
        full_metrics = _query_metrics(full[query_id], qrels[query_id], ks)
        compressed_metrics = _query_metrics(
            compressed[query_id], qrels[query_id], ks
        )
        if metric_names is None:
            metric_names = list(full_metrics)
        full_top10 = set(full[query_id][: min(10, depth)])
        compressed_top10 = set(compressed[query_id][: min(10, depth)])
        compressed_top100 = set(compressed[query_id][: min(100, depth)])
        top10_overlap = len(full_top10 & compressed_top10) / len(full_top10)
        retained = len(full_top10 & compressed_top100) / len(full_top10)
        fidelity["same_top1"].append(float(full[query_id][0] == compressed[query_id][0]))
        fidelity["top10_overlap"].append(top10_overlap)
        fidelity["full_top10_retained_at_compressed_top100"].append(retained)
        fidelity["full_top10_escape_compressed_top100"].append(1.0 - retained)
        doc_positions = {
            doc_id: position
            for position, doc_id in enumerate(
                dict.fromkeys(full[query_id][:depth] + compressed[query_id][:depth])
            )
        }
        fidelity["rbo_ext_at_100_p0.95"].append(
            rbo_ext(
                [doc_positions[doc_id] for doc_id in full[query_id][:depth]],
                [doc_positions[doc_id] for doc_id in compressed[query_id][:depth]],
                p=0.95,
            )
        )
        per_query.append(
            {
                "query_id": query_id,
                "positive_qrels": len(qrels[query_id]),
                "full": full_metrics,
                "compressed": compressed_metrics,
                "regret": {
                    name: full_metrics[name] - compressed_metrics[name]
                    for name in full_metrics
                },
                "fidelity": {
                    name: values[-1] for name, values in fidelity.items()
                },
            }
        )
    assert metric_names is not None

    quality: dict[str, Any] = {}
    for offset, name in enumerate(metric_names):
        reference = np.asarray([row["full"][name] for row in per_query])
        candidate = np.asarray([row["compressed"][name] for row in per_query])
        threshold = 0.05 if name == "recall_at_100" else 0.10
        summary = regret_summary(
            reference,
            candidate,
            catastrophic_threshold=threshold,
            seed=bootstrap_seed + offset,
            resamples=bootstrap_resamples,
        )
        regret = reference - candidate
        summary["improved_queries"] = int(np.sum(regret < -1e-12))
        summary["tied_queries"] = int(np.sum(np.abs(regret) <= 1e-12))
        summary["harmed_queries"] = int(np.sum(regret > 1e-12))
        quality[name] = summary

    fidelity_summary = {}
    for offset, (name, observations) in enumerate(fidelity.items()):
        values = np.asarray(observations)
        fidelity_summary[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "zero_queries": int(np.sum(values == 0.0)),
            "nonzero_queries": int(np.sum(values > 0.0)),
            "below_one_queries": int(np.sum(values < 1.0)),
            "bootstrap": bootstrap_mean_interval(
                values,
                seed=bootstrap_seed + 100 + offset,
                resamples=bootstrap_resamples,
            ),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "paired-official-omni-ranking-risk-2026-08-05",
        "regret_sign": "positive_means_compressed_worse_than_full",
        "qrels_used_by_compressor": False,
        "qrels_used_for_final_evaluation_only": True,
        "queries": len(query_ids),
        "ranking_depth": depth,
        "quality": quality,
        "ranking_fidelity": fidelity_summary,
        "per_query": per_query,
        "artifacts": {
            "qrels": {"path": str(qrels_path), "sha256": _sha256(qrels_path)},
            "full_ranking": {
                "path": str(full_ranking_path),
                "sha256": _sha256(full_ranking_path),
            },
            "compressed_ranking": {
                "path": str(compressed_ranking_path),
                "sha256": _sha256(compressed_ranking_path),
            },
        },
    }
    if depth >= 100:
        quality_gate = {
            "tolerance": 0.01,
            "ndcg_at_10_one_sided_95_upper_regret": quality["ndcg_at_10"][
                "bootstrap"
            ]["one_sided_95_upper"],
            "recall_at_100_one_sided_95_upper_regret": quality["recall_at_100"][
                "bootstrap"
            ]["one_sided_95_upper"],
        }
        quality_gate["passes"] = bool(
            quality_gate["ndcg_at_10_one_sided_95_upper_regret"]
            <= quality_gate["tolerance"]
            and quality_gate["recall_at_100_one_sided_95_upper_regret"]
            <= quality_gate["tolerance"]
        )
        ranking_gate = {
            "minimum_top10_mean": 0.90,
            "minimum_top10_one_sided_95_lower": 0.88,
            "minimum_full_top10_at_compressed100_one_sided_95_lower": 0.995,
            "observed_top10_mean": fidelity_summary["top10_overlap"]["mean"],
            "observed_top10_one_sided_95_lower": fidelity_summary[
                "top10_overlap"
            ]["bootstrap"]["one_sided_95_lower"],
            "observed_full_top10_at_compressed100_one_sided_95_lower": (
                fidelity_summary["full_top10_retained_at_compressed_top100"][
                    "bootstrap"
                ]["one_sided_95_lower"]
            ),
        }
        ranking_gate["passes"] = bool(
            ranking_gate["observed_top10_mean"]
            >= ranking_gate["minimum_top10_mean"]
            and ranking_gate["observed_top10_one_sided_95_lower"]
            >= ranking_gate["minimum_top10_one_sided_95_lower"]
            and ranking_gate[
                "observed_full_top10_at_compressed100_one_sided_95_lower"
            ]
            >= ranking_gate[
                "minimum_full_top10_at_compressed100_one_sided_95_lower"
            ]
        )
    else:
        quality_gate = {
            "passes": None,
            "status": "not_evaluated_ranking_depth_below_100",
        }
        ranking_gate = {
            "passes": None,
            "status": "not_evaluated_ranking_depth_below_100",
        }
    report["quality_safety_gate"] = quality_gate
    report["qrel_free_ranking_gate"] = ranking_gate

    for label, results_path in (
        ("full", full_results_path),
        ("compressed", compressed_results_path),
    ):
        if results_path is None:
            continue
        observed = {
            name: quality[name][f"{'reference' if label == 'full' else 'candidate'}_mean"]
            for name in metric_names
        }
        official = _official_metrics(results_path)
        mismatches = {
            name: {"observed": value, "official": official.get(name)}
            for name, value in observed.items()
            if name not in official or not math.isclose(value, official[name], abs_tol=1e-12)
        }
        if mismatches:
            raise ValueError(f"{label} metrics do not reproduce official results: {mismatches}")
        report["artifacts"][f"{label}_results"] = {
            "path": str(results_path),
            "sha256": _sha256(results_path),
            "official_metrics_reproduced": True,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--full-ranking", type=Path, required=True)
    parser.add_argument("--compressed-ranking", type=Path, required=True)
    parser.add_argument("--full-results", type=Path)
    parser.add_argument("--compressed-results", type=Path)
    parser.add_argument("--ranking-depth", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_pair(
        args.qrels,
        args.full_ranking,
        args.compressed_ranking,
        depth=args.ranking_depth,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        full_results_path=args.full_results,
        compressed_results_path=args.compressed_results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "queries": report["queries"],
                "quality": report["quality"],
                "ranking_fidelity": report["ranking_fidelity"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
