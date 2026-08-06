#!/usr/bin/env python3
"""Compare frozen Omni audits with pinned official ViDoRe pipeline results."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _omni_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    per_query = report["per_query"]
    methods = sorted(per_query[0]["ndcg_at_10"])
    ndcg = {
        method: statistics.fmean(row["ndcg_at_10"][method] for row in per_query)
        for method in methods
    }
    candidate_recall = {
        method: statistics.fmean(
            row["candidate_recall_at_100"][method] for row in per_query
        )
        for method in sorted(per_query[0]["candidate_recall_at_100"])
    }
    return {
        "artifact": {"path": str(path), "sha256": _sha256(path)},
        "queries": report["analysis_scope"]["queries"],
        "corpus_pages": report["analysis_scope"]["corpus_pages"],
        "mean_ndcg_at_10": ndcg,
        "mean_qrel_candidate_recall_at_100": candidate_recall,
        "index_bytes": {
            key.removesuffix("_index_bytes"): value
            for key, value in report["multi_locator_cost_ledger"].items()
            if key.endswith("_index_bytes")
        },
    }


def compare(
    official_root: Path,
    official_revision: str,
    official_models: list[str],
    omni_reports: dict[str, tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    if not official_models:
        raise ValueError("at least one official model is required")
    if not omni_reports:
        raise ValueError("at least one Omni report is required")

    domains: dict[str, Any] = {}
    for domain, (omni_path, omni_report) in sorted(omni_reports.items()):
        official_rows = []
        official_filename = f"vidore_v3_{domain}.json"
        for model in official_models:
            result_path = official_root / "results" / "metrics" / model / official_filename
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            result = _load(result_path)["aggregated_metrics"]
            overall = result["overall"]
            timing = result.get("timing", {})
            official_rows.append(
                {
                    "model": model,
                    "artifact": {
                        "path": str(result_path.relative_to(official_root)),
                        "sha256": _sha256(result_path),
                    },
                    "ndcg_at_10": overall["ndcg_cut_10"],
                    "recall_at_100": overall["recall_100"],
                    "indexing_ms_per_document": timing.get(
                        "indexing_throughput_ms_per_doc"
                    ),
                    "search_ms_per_query": timing.get(
                        "search_throughput_ms_per_query"
                    ),
                }
            )
        domains[domain] = {
            "omni": _omni_summary(omni_path, omni_report),
            "official_pipelines": official_rows,
        }

    return {
        "protocol": "omni_vs_pinned_official_vidore_v3_pipelines_2026-08-06",
        "official_source": {
            "repository": "https://github.com/illuin-tech/vidore-benchmark",
            "revision": official_revision,
        },
        "comparison_boundary": {
            "quality": "direct comparison on the same public domain split",
            "timing": (
                "official reported hardware/configuration context only; not directly "
                "comparable with the local Omni A100 measurements"
            ),
            "omni_recall": (
                "candidate recall is available for locator methods; Full and cascades "
                "require the frozen evaluator summaries for end-to-end Recall@100"
            ),
        },
        "domains": domains,
    }


def _parse_specifications(specifications: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for specification in specifications:
        name, separator, path_text = specification.partition("=")
        if not separator or not name or not path_text or name in parsed:
            raise ValueError(f"invalid or duplicate NAME=PATH: {specification}")
        parsed[name] = Path(path_text)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--official-revision", required=True)
    parser.add_argument("--official-model", action="append", required=True)
    parser.add_argument("--omni-report", action="append", required=True, help="DOMAIN=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report_paths = _parse_specifications(args.omni_report)
    reports = {name: (path, _load(path)) for name, path in report_paths.items()}
    result = compare(
        args.official_root,
        args.official_revision,
        args.official_model,
        reports,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"domains": sorted(result["domains"])}, sort_keys=True))


if __name__ == "__main__":
    main()
