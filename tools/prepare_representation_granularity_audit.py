#!/usr/bin/env python3
"""Prepare frozen whole-page and sub-page OmniColPress diagnostic corpora."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from reprforge.representation_granularity import (
    deterministic_neutral_order,
    fixed_quadrants,
    xycut_regions,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def select_pages(config: dict[str, Any], atlas: dict[str, Any], domain: str) -> list[dict[str, Any]]:
    cohort = config["cohort"]
    by_category = {
        category: [row for row in atlas["page_values"] if row["category"] == category]
        for category in ("positive", "negative", "neutral")
    }
    by_category["positive"].sort(key=lambda row: (-row["net_mean_ndcg_delta"], row["doc_id"]))
    by_category["negative"].sort(key=lambda row: (row["net_mean_ndcg_delta"], row["doc_id"]))
    neutral_ids = deterministic_neutral_order(
        (row["doc_id"] for row in by_category["neutral"]),
        protocol_id=config["protocol_id"],
        domain=domain,
    )
    neutral_by_id = {str(row["doc_id"]): row for row in by_category["neutral"]}
    selected = (
        by_category["positive"][: int(cohort["positive_pages"])]
        + by_category["negative"][: int(cohort["negative_pages"])]
        + [neutral_by_id[doc_id] for doc_id in neutral_ids[: int(cohort["neutral_pages"])]]
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    atlas = json.loads(args.atlas.read_text(encoding="utf-8"))
    selected = select_pages(config, atlas, args.domain)
    corpus = {str(row["docid"]): row for row in read_jsonl(args.dataset_root / "corpus.jsonl")}
    queries = read_jsonl(args.dataset_root / "queries.jsonl")
    qrels = read_jsonl(args.dataset_root / "qrels.jsonl")
    categories = {str(row["doc_id"]): row["category"] for row in selected}
    selected_ids = set(categories)

    organizations = config["organizations"]
    manifests: dict[str, Any] = {}
    for organization in organizations:
        root = args.output_root / organization
        assets = root / "assets"
        assets.mkdir(parents=True)
        rows: list[dict[str, Any]] = []
        units: list[dict[str, Any]] = []
        for parent_id in sorted(selected_ids, key=lambda value: int(value)):
            source = corpus[parent_id]
            with Image.open(args.dataset_root / "assets" / str(source["image"])) as opened:
                image = opened.convert("RGB")
                if organization == "whole_page":
                    boxes = [(0, 0, image.width, image.height)]
                elif organization == "fixed_quadrants":
                    boxes = fixed_quadrants(
                        image.width,
                        image.height,
                        overlap_fraction=float(organizations[organization]["overlap_fraction"]),
                    )
                else:
                    spec = organizations[organization]
                    boxes = xycut_regions(
                        image,
                        maximum_units=int(spec["maximum_units_per_page"]),
                        analysis_maximum_side=int(spec["analysis_maximum_side"]),
                        ink_threshold=int(spec["ink_threshold"]),
                        minimum_region_fraction=float(spec["minimum_region_fraction"]),
                        minimum_whitespace_gap_fraction=float(spec["minimum_whitespace_gap_fraction"]),
                        crop_padding_fraction=float(spec["crop_padding_fraction"]),
                    )
                for index, box in enumerate(boxes):
                    unit_id = f"{parent_id}--{organization}--{index}"
                    filename = f"{unit_id}.png"
                    image.crop(box).save(assets / filename, format="PNG")
                    rows.append({"docid": unit_id, "image": filename, "text": str(source.get("text", ""))})
                    units.append(
                        {
                            "unit_id": unit_id,
                            "parent_id": parent_id,
                            "category": categories[parent_id],
                            "bbox": list(box),
                            "parent_size": [image.width, image.height],
                        }
                    )
        write_jsonl(root / "corpus.jsonl", rows)
        write_jsonl(root / "queries.jsonl", queries)
        derived_qrels = []
        relevant_parents: dict[str, list[dict[str, Any]]] = {}
        for row in qrels:
            parent = str(row["doc_id"])
            if parent in selected_ids:
                relevant_parents.setdefault(parent, []).append(row)
        units_by_parent: dict[str, list[str]] = {}
        for unit in units:
            units_by_parent.setdefault(unit["parent_id"], []).append(unit["unit_id"])
        for parent, parent_qrels in relevant_parents.items():
            for row in parent_qrels:
                for unit_id in units_by_parent[parent]:
                    derived_qrels.append(
                        {"doc_id": unit_id, "query_id": str(row["query_id"]), "relevance": row["relevance"]}
                    )
        write_jsonl(root / "qrels.jsonl", derived_qrels)
        manifest = {
            "protocol_id": config["protocol_id"],
            "domain": args.domain,
            "organization": organization,
            "parents": len(selected_ids),
            "units": units,
            "category_counts": {
                category: sum(value == category for value in categories.values())
                for category in ("positive", "negative", "neutral")
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifests[organization] = {"parents": len(selected_ids), "units": len(units)}
    shutil.copy2(args.dataset_root / "qrels.jsonl", args.output_root / "parent-qrels.jsonl")
    (args.output_root / "summary.json").write_text(
        json.dumps({"protocol_id": config["protocol_id"], "domain": args.domain, "organizations": manifests}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
