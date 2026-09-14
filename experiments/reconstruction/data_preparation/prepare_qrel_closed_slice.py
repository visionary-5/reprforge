#!/usr/bin/env python3
"""Create a deterministic qrel-closed page pool from an existing sealed slice."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--query-count", type=int, required=True)
    parser.add_argument("--target-pages", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")

    import pyarrow as pa
    import pyarrow.parquet as pq

    items = read_jsonl(args.route_root / "items.jsonl")
    queries = read_jsonl(args.route_root / "queries.jsonl")[: args.query_count]
    item_to_index = {str(row["item_id"]): index for index, row in enumerate(items)}
    relevant_ids = sorted(
        {str(item_id) for query in queries for item_id in query["relevance"]}
    )
    missing = [item_id for item_id in relevant_ids if item_id not in item_to_index]
    if missing:
        raise ValueError(f"route is not qrel closed: {missing[:5]}")
    if len(relevant_ids) > args.target_pages:
        raise ValueError("target pool is smaller than the relevant-page union")
    relevant_indices = {item_to_index[item_id] for item_id in relevant_ids}
    remaining = [index for index in range(len(items)) if index not in relevant_indices]
    rng = random.Random(args.seed)
    distractors = rng.sample(remaining, args.target_pages - len(relevant_indices))
    selected_indices = sorted(relevant_indices | set(distractors))

    table = pq.read_table(args.slice)
    selected_table = table.take(pa.array(selected_indices))
    args.output_root.mkdir(parents=True)
    route_output = args.output_root / "route"
    route_output.mkdir()
    pq.write_table(selected_table, args.output_root / "slice.parquet")
    (route_output / "items.jsonl").write_text(
        "".join(json.dumps(items[index]) + "\n" for index in selected_indices)
    )
    (route_output / "queries.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in queries)
    )
    manifest = {
        "query_count": args.query_count,
        "target_pages": args.target_pages,
        "relevant_pages": len(relevant_indices),
        "distractor_pages": len(distractors),
        "seed": args.seed,
        "selected_source_indices": selected_indices,
        "all_query_qrels_present": all(
            set(map(str, row["relevance"])).issubset(
                {str(items[index]["item_id"]) for index in selected_indices}
            )
            for row in queries
        ),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
