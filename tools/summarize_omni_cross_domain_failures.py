#!/usr/bin/env python3
"""Summarize frozen Omni P1 failure audits across domains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _best_marker(target: dict[str, Any], *, requires_agc: bool) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in target.get("markers", [])
            if bool(row["requires_agc"]) == requires_agc
        ),
        None,
    )


def summarize(reports: dict[str, tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("cross-domain summary requires at least two domains")
    domains: dict[str, Any] = {}
    for name, (path, report) in reports.items():
        scope = report["analysis_scope"]
        mechanisms = report["failure_mechanism_summary"]
        escape = report["escape_audit"]
        boundary = report["relevant_candidate_boundary"]
        signals = report["observable_risk_signal_audit"]["targets"]
        domains[name] = {
            "artifact": {"path": str(path), "sha256": _sha256(path)},
            "queries": scope["queries"],
            "corpus_pages": scope["corpus_pages"],
            "teacher_containment_counts": report["teacher_containment_groups"]["counts"],
            "quality_preservation_counts": report["quality_preservation_groups"]["counts"],
            "hpool_quality_harm_queries": mechanisms["hpool_quality_harm_queries"],
            "hpool_harm_with_full_top10_contained_at_100": mechanisms[
                "hpool_quality_harm_with_full_top10_contained_at_100"
            ],
            "hpool_harm_recovered_by_cascade100": mechanisms[
                "hpool_quality_harm_recovered_by_cascade100"
            ],
            "ranking_escape_queries": escape["ranking_escape_queries"],
            "teacher_evidence_escape_queries": escape[
                "teacher_evidence_escape_queries"
            ],
            "teacher_evidence_escape_documents": escape[
                "teacher_evidence_escape_documents"
            ],
            "evidence_escape_documents_recovered_by_agc": escape[
                "teacher_evidence_escape_documents_recovered_by_agc_at_100"
            ],
            "hpool_relevant_recall_at_100": boundary["hpool_mean_recall_at_100"],
            "union_relevant_recall_at_100": boundary["union_mean_recall_at_100"],
            "queries_improved_by_union": boundary["queries_improved_by_union"],
            "risk_markers": {
                target_name: {
                    "positives": target["positives"],
                    "best_free": _best_marker(target, requires_agc=False),
                    "best_with_agc": _best_marker(target, requires_agc=True),
                    "unavailable_reason": target.get("unavailable_reason"),
                }
                for target_name, target in signals.items()
            },
        }

    total_queries = sum(row["queries"] for row in domains.values())
    total_harm = sum(row["hpool_quality_harm_queries"] for row in domains.values())
    total_contained_harm = sum(
        row["hpool_harm_with_full_top10_contained_at_100"]
        for row in domains.values()
    )
    total_recovered = sum(
        row["hpool_harm_recovered_by_cascade100"] for row in domains.values()
    )
    return {
        "protocol": "frozen_cross_domain_omni_failure_summary_2026-08-05",
        "domains": domains,
        "aggregate": {
            "domains": len(domains),
            "queries": total_queries,
            "hpool_quality_harm_queries": total_harm,
            "hpool_harm_with_full_top10_contained_at_100": total_contained_harm,
            "hpool_harm_recovered_by_cascade100": total_recovered,
            "contained_fraction_among_hpool_harms": (
                total_contained_harm / total_harm if total_harm else None
            ),
            "cascade100_recovery_fraction_among_hpool_harms": (
                total_recovered / total_harm if total_harm else None
            ),
            "ranking_escape_queries": sum(
                row["ranking_escape_queries"] for row in domains.values()
            ),
            "teacher_evidence_escape_queries": sum(
                row["teacher_evidence_escape_queries"] for row in domains.values()
            ),
            "teacher_evidence_escape_documents": sum(
                row["teacher_evidence_escape_documents"] for row in domains.values()
            ),
            "evidence_escape_documents_recovered_by_agc": sum(
                row["evidence_escape_documents_recovered_by_agc"]
                for row in domains.values()
            ),
            "queries_improved_by_union": sum(
                row["queries_improved_by_union"] for row in domains.values()
            ),
            "query_weighted_hpool_relevant_recall_at_100": sum(
                row["queries"] * row["hpool_relevant_recall_at_100"]
                for row in domains.values()
            )
            / total_queries,
            "query_weighted_union_relevant_recall_at_100": sum(
                row["queries"] * row["union_relevant_recall_at_100"]
                for row in domains.values()
            )
            / total_queries,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for specification in args.report:
        name, separator, path_text = specification.partition("=")
        if not separator or not name or not path_text or name in reports:
            raise ValueError(f"invalid or duplicate NAME=PATH: {specification}")
        path = Path(path_text)
        reports[name] = (path, json.loads(path.read_text(encoding="utf-8")))
    result = summarize(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
