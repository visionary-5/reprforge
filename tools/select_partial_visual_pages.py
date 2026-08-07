#!/usr/bin/env python3
"""Build nested qrel-free corpus subsets for physical visual indexing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from reprforge.partial_page_selector import (
    PageRiskFeatures,
    budget_count,
    selection_order,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _image_features(path: Path, maximum_side: int) -> tuple[float, float, float]:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        scale = min(1.0, maximum_side / max(grayscale.size))
        if scale < 1.0:
            grayscale = grayscale.resize(
                tuple(max(1, round(value * scale)) for value in grayscale.size),
                Image.Resampling.BILINEAR,
            )
        values = np.asarray(grayscale, dtype=np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / values.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum() / 8.0)
    horizontal = (
        float(np.abs(np.diff(values.astype(np.float32), axis=1)).mean() / 255.0)
        if values.shape[1] > 1
        else 0.0
    )
    vertical = (
        float(np.abs(np.diff(values.astype(np.float32), axis=0)).mean() / 255.0)
        if values.shape[0] > 1
        else 0.0
    )
    edge_energy = 0.5 * (horizontal + vertical)
    nonwhite_fraction = float(np.mean(values < 245))
    return entropy, edge_energy, nonwhite_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strategy", default="risk_cover_round_robin")
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = _read_jsonl(args.corpus)
    if not rows:
        raise ValueError("corpus is empty")
    maximum_side = int(config["selector"]["maximum_image_side"])
    features: list[PageRiskFeatures] = []
    for row in rows:
        doc_id = str(row["docid"])
        image_name = str(row["image"])
        entropy, edge_energy, nonwhite_fraction = _image_features(
            args.assets / image_name, maximum_side
        )
        features.append(
            PageRiskFeatures(
                doc_id=doc_id,
                text_chars=len(str(row.get("text") or "").strip()),
                grayscale_entropy=entropy,
                edge_energy=edge_energy,
                nonwhite_fraction=nonwhite_fraction,
            )
        )
    order = selection_order(features, strategy=args.strategy, seed=args.seed)
    row_by_id = {str(row["docid"]): row for row in rows}
    feature_by_id = {row.doc_id: row for row in features}
    args.output_root.mkdir(parents=True)
    feature_path = args.output_root / "features.jsonl"
    with feature_path.open("w", encoding="utf-8") as handle:
        for doc_id in order:
            feature = feature_by_id[doc_id]
            handle.write(json.dumps(feature.__dict__, sort_keys=True) + "\n")
    for budget in map(float, config["budgets"]):
        count = budget_count(len(rows), budget)
        selected_ids = order[:count]
        budget_root = args.output_root / f"budget-{round(100 * budget):03d}"
        budget_root.mkdir()
        corpus_path = budget_root / "corpus.jsonl"
        with corpus_path.open("w", encoding="utf-8") as handle:
            for doc_id in selected_ids:
                handle.write(json.dumps(row_by_id[doc_id], sort_keys=True) + "\n")
        manifest = {
            "schema_version": 1,
            "protocol_id": config["protocol_id"],
            "strategy": args.strategy,
            "seed": args.seed,
            "budget_fraction": budget,
            "source_pages": len(rows),
            "selected_pages": count,
            "selected_fraction": count / len(rows),
            "selected_doc_ids": selected_ids,
            "information_boundary": {
                "uses_qrels": False,
                "uses_complete_visual_scores": False,
                "features": config["selector"]["inputs"],
            },
            "sha256": {
                "config": _sha(args.config),
                "source_corpus": _sha(args.corpus),
                "selected_corpus": _sha(corpus_path),
                "features": _sha(feature_path),
            },
        }
        (budget_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"pages": len(rows), "strategy": args.strategy, "budgets": config["budgets"]}))


if __name__ == "__main__":
    main()
