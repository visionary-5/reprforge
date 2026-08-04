#!/usr/bin/env python3
"""Summarize pre-qrel gate decisions against post-certificate safety."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reprforge.compression_gate_transfer import summarize_gate_transfer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("CERTIFICATE", "EVALUATION"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    artifact_pairs = []
    for certificate_value, evaluation_value in args.pair:
        certificate_path = Path(certificate_value)
        evaluation_path = Path(evaluation_value)
        certificate = json.loads(certificate_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
        if certificate.get("stage") != "pre-qrel-ranking-certification":
            raise ValueError(f"{certificate_path} is not a pre-qrel certificate")
        if certificate.get("qrels_loaded") is not False:
            raise ValueError(f"{certificate_path} does not attest qrels_loaded=false")
        for key in ("dataset", "candidate"):
            if certificate.get(key) != evaluation.get(key):
                raise ValueError(f"{key} differs across {certificate_path} and {evaluation_path}")
        certificate_artifacts = certificate.get("artifacts", {})
        evaluation_artifacts = evaluation.get("artifacts", {})
        for key in ("reference_runtime_sha256", "candidate_runtime_sha256"):
            if certificate_artifacts.get(key) != evaluation_artifacts.get(key):
                raise ValueError(f"{key} differs across certificate and evaluation")
        records.append(
            {
                "dataset": evaluation["dataset"],
                "candidate": evaluation["candidate"],
                "certificate_passes": certificate[
                    "qrel_free_ranking_certificate"
                ]["passes"],
                "safety_passes": evaluation["safety_gate"]["passes"],
                "resident_fraction": evaluation["cost"]["vector_byte_fraction"],
                "mean_ndcg10_regret": evaluation["quality"]["ndcg_at_10"][
                    "mean_regret"
                ],
            }
        )
        artifact_pairs.append(
            {
                "certificate": str(certificate_path),
                "evaluation": str(evaluation_path),
            }
        )
    report = summarize_gate_transfer(records)
    report.update(
        {
            "schema_version": 1,
            "protocol": "qrel-free-compression-risk-2026-08-04",
            "artifact_pairs": artifact_pairs,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
