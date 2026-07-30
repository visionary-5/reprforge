#!/usr/bin/env python3
"""Merge compatible MMDocIR route score banks without re-encoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from reprforge.policy_replay import (
    evaluate_plan,
    fixed_hybrid_plan,
    load_replay_data,
    uniform_plan,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_jsonl(inputs: Sequence[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for path in inputs:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        target.write(line.rstrip("\n") + "\n")


def merge_replay_banks(inputs: Sequence[Path], output: Path) -> dict:
    if len(inputs) < 2:
        raise ValueError("at least two replay banks are required")
    if output.exists():
        raise FileExistsError(f"merged bank already exists: {output}")
    manifests = [
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for path in inputs
    ]
    contracts = {
        (
            str(manifest["dataset_revision"]),
            str(manifest["model_revision"]),
        )
        for manifest in manifests
    }
    if len(contracts) != 1:
        raise ValueError("input banks have different dataset/model revisions")
    if any(manifest.get("embeddings_persisted") for manifest in manifests):
        raise ValueError(
            "merge physical embedding banks with merge_embedding_banks instead"
        )

    output.mkdir(parents=True)
    for filename in ("items.jsonl", "queries.jsonl", "scores.jsonl"):
        _merge_jsonl(
            [path / filename for path in inputs],
            output / filename,
        )
    data = load_replay_data(
        output / "items.jsonl",
        output / "queries.jsonl",
        output / "scores.jsonl",
    )
    plans = {
        f"uniform-{route}": uniform_plan(data.items, route)
        for route in data.routes
    }
    for route in data.routes:
        if route == "text":
            continue
        plans[f"fixed-hybrid-{route}"] = fixed_hybrid_plan(
            data.items,
            image_route=route,
        )
    baselines = {
        name: evaluate_plan(data, plan, ks=(1, 5, 10))
        for name, plan in plans.items()
    }
    (output / "baselines.json").write_text(
        json.dumps(baselines, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    document_indices = []
    domains = set()
    for manifest in manifests:
        for document in manifest["documents"]:
            document_indices.append(int(document["document_index"]))
            domains.add(str(document.get("domain") or ""))
    if len(set(document_indices)) != len(document_indices):
        raise ValueError("input banks contain duplicate documents")
    dataset_revision, model_revision = contracts.pop()
    manifest = {
        "dataset": manifests[0]["dataset"],
        "dataset_revision": dataset_revision,
        "model": manifests[0]["model"],
        "model_revision": model_revision,
        "merge_inputs": [
            {
                "path": str(path),
                "manifest_sha256": _sha256(path / "manifest.json"),
            }
            for path in inputs
        ],
        "document_indices": sorted(document_indices),
        "document_count": len(document_indices),
        "domains": sorted(domains),
        "layout_count": len(data.items),
        "query_count": len(data.queries),
        "routes": list(data.routes),
        "embeddings_persisted": False,
        "artifacts": {
            filename: _sha256(output / filename)
            for filename in ("items.jsonl", "queries.jsonl", "scores.jsonl")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        dest="inputs",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = merge_replay_banks(args.inputs, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
