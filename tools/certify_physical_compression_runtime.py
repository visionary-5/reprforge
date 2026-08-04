#!/usr/bin/env python3
"""Certify a frozen physical runtime on reserved queries without qrels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.compression_risk_metrics import (
    ranking_fidelity,
    ranking_safety_certificate,
)


def _sha256(artifact: Path) -> str:
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(archive, key: str) -> list[str]:
    return [str(value) for value in archive[key]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-runtime", type=Path, required=True)
    parser.add_argument("--candidate-runtime", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    reference = np.load(args.reference_runtime, allow_pickle=False)
    candidate = np.load(args.candidate_runtime, allow_pickle=False)
    plan = json.loads(args.plan.read_text())
    if plan.get("stage") != "pre-qrel-physical-plan":
        raise ValueError("plan is not a pre-qrel physical plan")
    if plan.get("qrels_loaded") is not False:
        raise ValueError("plan does not attest qrels_loaded=false")
    if plan["artifacts"]["full_runtime_sha256"] != _sha256(
        args.reference_runtime
    ):
        raise ValueError("reference runtime hash differs from the plan")
    query_ids = _ids(reference, "query_ids")
    if query_ids != _ids(candidate, "query_ids") or _ids(
        reference, "corpus_ids"
    ) != _ids(candidate, "corpus_ids"):
        raise ValueError("reference and candidate IDs differ")
    roles = np.asarray(plan["query_split_roles"])
    if roles.shape != (len(query_ids),):
        raise ValueError("plan query roles do not align with the runtime")
    evaluation = np.flatnonzero(roles == "eval")
    if not len(evaluation):
        raise ValueError("plan reserves no evaluation queries")
    fidelity = ranking_fidelity(
        reference["scores"][evaluation], candidate["scores"][evaluation]
    )
    certificate = ranking_safety_certificate(
        fidelity,
        seed=args.seed,
        resamples=4000,
    )
    report = {
        "schema_version": 1,
        "stage": "pre-qrel-physical-runtime-certificate",
        "protocol": "qrel-free-physical-compression-crossfit-v1",
        "dataset": plan["dataset"],
        "qrels_loaded": False,
        "qrels_used": False,
        "evaluation_queries": len(evaluation),
        "selected_state": plan["selected_state"],
        "ranking_certificate": certificate,
        "ranking_fidelity": {
            name: {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p05": float(np.quantile(values, 0.05)),
            }
            for name, values in fidelity.items()
        },
        "artifacts": {
            "reference_runtime_sha256": _sha256(args.reference_runtime),
            "candidate_runtime_sha256": _sha256(args.candidate_runtime),
            "plan_sha256": _sha256(args.plan),
            "materialization_sha256": _sha256(args.materialization),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
