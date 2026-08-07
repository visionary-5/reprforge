#!/usr/bin/env python3
"""Build a signed page-level VLM representation value atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from reprforge.residual_materialization_oracle import (
    auc,
    singleton_page_value_atlas,
)
from tools.analyze_residual_materialization_oracle import _surface
from tools.run_dvi_page_verifier_pilot import (
    _bm25_rankings,
    _load_qrels,
    _load_visual_ranking,
    _read_jsonl,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _features(path: Path, doc_ids: list[str]) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    by_id = {str(row["doc_id"]): row for row in rows}
    if set(by_id) != set(doc_ids):
        raise ValueError("feature IDs differ from corpus")
    return {
        "ocr_text_scarcity": np.asarray([-float(by_id[d]["text_chars"]) for d in doc_ids]),
        "grayscale_entropy": np.asarray([float(by_id[d]["grayscale_entropy"]) for d in doc_ids]),
        "edge_energy": np.asarray([float(by_id[d]["edge_energy"]) for d in doc_ids]),
        "nonwhite_fraction": np.asarray([float(by_id[d]["nonwhite_fraction"]) for d in doc_ids]),
    }


def _rank_frequency(order: np.ndarray, pages: int) -> np.ndarray:
    values = np.zeros(pages, dtype=np.float64)
    weights = 1.0 / np.log2(np.arange(2, order.shape[1] + 2, dtype=np.float64))
    for row in order:
        values[row] += weights
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--colsmol-ranking", type=Path, required=True)
    parser.add_argument("--omni-ranking", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    retrieval = config["retrieval"]
    depth = int(retrieval["rank_depth"])
    corpus = _read_jsonl(args.dataset_root / "corpus.jsonl")
    queries = _read_jsonl(args.dataset_root / "queries.jsonl")
    qrels = _load_qrels(args.dataset_root / "qrels.jsonl")
    bm25, bm25_cost = _bm25_rankings(corpus, queries, depth)
    colsmol = _load_visual_ranking(args.colsmol_ranking, depth)
    omni = _load_visual_ranking(args.omni_ranking, depth)
    surface = _surface(
        name=args.domain,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        bm25=bm25,
        colsmol=colsmol,
        omni=omni,
        depth=depth,
    )
    atlas = singleton_page_value_atlas(
        surface,
        np.arange(surface.queries),
        rrf_constant=int(retrieval["rrf_constant"]),
        candidate_escape_depth=int(retrieval["candidate_escape_depth"]),
    )
    page_values = atlas["page_values"]
    positive = np.asarray([row["category"] == "positive" for row in page_values])
    negative = np.asarray([row["category"] == "negative" for row in page_values])
    net = np.asarray([row["net_mean_ndcg_delta"] for row in page_values])
    signals = _features(args.features, surface.doc_ids)
    signals.update(
        {
            "bm25_top100_frequency": _rank_frequency(surface.bm25, surface.pages),
            "colsmol_top100_frequency": _rank_frequency(surface.colsmol, surface.pages),
            "omni_top100_frequency_oracle": _rank_frequency(surface.omni, surface.pages),
            "qrel_frequency_oracle": np.sum(surface.qrels > 0, axis=0).astype(float),
        }
    )
    diagnostics = {}
    for name, values in signals.items():
        diagnostics[name] = {
            "auc_positive": auc(values, positive),
            "auc_negative": auc(values, negative),
            "pearson_with_net_value": (
                float(np.corrcoef(values, net)[0, 1])
                if float(np.std(values)) > 0 and float(np.std(net)) > 0
                else None
            ),
        }
    concentration = {}
    positive_mass = float(atlas["summary"]["positive_singleton_value_mass"])
    ordered = atlas["positive_order"]
    for fraction in map(float, config["concentration_budgets"]):
        count = min(surface.pages, max(1, math.ceil(fraction * surface.pages)))
        selected = ordered[:count]
        captured = float(np.sum(net[selected])) if selected else 0.0
        concentration[str(fraction)] = {
            "page_budget": count,
            "positive_pages_selected": len(selected),
            "positive_value_mass_recovery": captured / positive_mass if positive_mass else None,
        }
    gate = config["gate"]
    checks = {
        "positive_pages_are_sparse": atlas["summary"]["positive_page_fraction"]
        <= float(gate["maximum_positive_page_fraction"]),
        "positive_value_is_concentrated": concentration["0.05"][
            "positive_value_mass_recovery"
        ]
        >= float(gate["minimum_positive_value_mass_in_top_5pct_pages"]),
        "negative_pages_exist": atlas["summary"]["negative_page_fraction"]
        >= float(gate["minimum_negative_page_fraction"]),
    }
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "domain": args.domain,
        "dataset": {
            "pages": surface.pages,
            "queries": surface.queries,
            "sha256": {
                "corpus": _sha(args.dataset_root / "corpus.jsonl"),
                "queries": _sha(args.dataset_root / "queries.jsonl"),
                "qrels": _sha(args.dataset_root / "qrels.jsonl"),
                "colsmol_ranking": _sha(args.colsmol_ranking),
                "omni_ranking": _sha(args.omni_ranking),
                "features": _sha(args.features),
            },
        },
        "bm25_cost": bm25_cost,
        "summary": atlas["summary"],
        "concentration": concentration,
        "signal_diagnostics": diagnostics,
        "gate": {"checks": checks, "passes_page_value_sparsity": all(checks.values())},
        "page_values": [
            {**row, "doc_id": surface.doc_ids[int(row["page"])]}
            for row in page_values
        ],
        "warnings": [
            retrieval["warning"],
            config["information_boundary"]["future_qrels_and_omni_ranks"],
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"domain": args.domain, "summary": result["summary"], "concentration": result["concentration"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
