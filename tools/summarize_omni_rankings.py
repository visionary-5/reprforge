#!/usr/bin/env python3
"""Summarize a variable-depth OmniColPress ranking matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.compression_risk_metrics import regret_summary
from tools.analyze_omni_pair import _query_metrics, load_qrels, load_ranking


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ranking_depth(path: Path) -> int:
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"expected 3 tab fields at {path}:{line_number}")
            counts[fields[0]] = counts.get(fields[0], 0) + 1
    depths = set(counts.values())
    if len(depths) != 1:
        raise ValueError(f"ranking has nonuniform query depths: {depths}")
    return depths.pop()


def summarize_rankings(
    qrels_path: Path,
    ranking_paths: dict[str, Path],
    *,
    reference: str,
    bootstrap_seed: int = 20260805,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    if reference not in ranking_paths:
        raise ValueError("reference must name one supplied ranking")
    qrels = load_qrels(qrels_path)
    query_ids = sorted(qrels, key=lambda value: (not value.isdigit(), value))
    observations: dict[str, dict[str, np.ndarray]] = {}
    rankings: dict[str, Any] = {}
    for name, path in ranking_paths.items():
        depth = _ranking_depth(path)
        ranking = load_ranking(path, expected_depth=depth)
        if set(ranking) != set(qrels):
            raise ValueError(f"{name} ranking and qrel query IDs differ")
        ks = tuple(sorted(set(k for k in (1, 5, 10, depth) if k <= depth)))
        rows = [_query_metrics(ranking[qid], qrels[qid], ks) for qid in query_ids]
        metrics = {
            metric: np.asarray([row[metric] for row in rows], dtype=np.float64)
            for metric in rows[0]
        }
        metrics[f"mrr_at_{depth}"] = metrics.pop("mrr_at_100")
        observations[name] = metrics
        rankings[name] = {
            "depth": depth,
            "metrics": {metric: float(values.mean()) for metric, values in metrics.items()},
            "artifact": {"path": str(path), "sha256": _sha256(path)},
        }

    reference_metrics = observations[reference]
    paired_regret: dict[str, Any] = {}
    for candidate_offset, (name, metrics) in enumerate(observations.items()):
        if name == reference:
            continue
        shared = sorted(set(reference_metrics) & set(metrics))
        paired_regret[name] = {}
        for metric_offset, metric in enumerate(shared):
            paired_regret[name][metric] = regret_summary(
                reference_metrics[metric],
                metrics[metric],
                catastrophic_threshold=(
                    0.05 if metric == "recall_at_100" else 0.10
                ),
                seed=bootstrap_seed + candidate_offset * 100 + metric_offset,
                resamples=bootstrap_resamples,
            )
    return {
        "schema_version": 1,
        "protocol": "official-omni-variable-depth-matrix-2026-08-05",
        "reference": reference,
        "queries": len(query_ids),
        "qrels_used_for_final_evaluation_only": True,
        "rankings": rankings,
        "paired_regret": paired_regret,
        "qrels": {"path": str(qrels_path), "sha256": _sha256(qrels_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument(
        "--ranking",
        action="append",
        required=True,
        help="repeat NAME=PATH",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranking_paths = {}
    for specification in args.ranking:
        name, separator, path = specification.partition("=")
        if not separator or not name or not path or name in ranking_paths:
            raise ValueError(f"invalid or duplicate NAME=PATH: {specification}")
        ranking_paths[name] = Path(path)
    report = summarize_rankings(
        args.qrels,
        ranking_paths,
        reference=args.reference,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["rankings"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
