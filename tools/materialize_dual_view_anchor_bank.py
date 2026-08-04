#!/usr/bin/env python3
"""Materialize selected full-document anchors beside a cheap cover bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.heterogeneous_index import _load_shard, write_embedding_bank


def _sha256(artifact: Path) -> str:
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _views(vectors: np.ndarray, offsets: np.ndarray, positions: np.ndarray):
    return tuple(
        np.asarray(vectors[int(offsets[index]) : int(offsets[index + 1])])
        for index in positions
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-bank", type=Path, required=True)
    parser.add_argument("--cheap-bank", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--policy", default="boundary_pareto")
    parser.add_argument("--point-index", type=int, default=0)
    parser.add_argument("--full-route", default="image")
    parser.add_argument("--cheap-route", default="image-pool-4")
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    if plan.get("qrels_used_by_compiler") is not False:
        raise ValueError("plan does not attest qrels_used_by_compiler=false")
    if plan.get("stage") == "pre-qrel-physical-plan":
        if plan.get("qrels_loaded") is not False:
            raise ValueError("strict plan does not attest qrels_loaded=false")
        if plan.get("selected_state") != "dual_view":
            raise ValueError("strict plan abstained to full and needs no anchor bank")
        if plan.get("policy") != args.policy:
            raise ValueError("requested policy differs from the strict plan")
        positions = np.asarray(plan["upgraded_documents"], dtype=np.int64)
        point_index = None
    else:
        points = plan["policies"][args.policy]["residual_affine"]
        if not 0 <= args.point_index < len(points):
            raise ValueError("point index lies outside the selected policy curve")
        point = points[args.point_index]
        positions = np.asarray(point["upgraded_documents"], dtype=np.int64)
        point_index = args.point_index
    if not len(positions) or len(np.unique(positions)) != len(positions):
        raise ValueError("plan must select unique full anchors")

    full_ids, full_vectors, full_offsets = _load_shard(
        args.full_bank / "routes" / args.full_route
    )
    cheap_ids, _, _ = _load_shard(args.cheap_bank / "routes" / args.cheap_route)
    if full_ids != cheap_ids or len(full_ids) != int(plan["corpus"]):
        raise ValueError("full/cheap bank IDs do not align with the plan corpus")
    if np.any(positions < 0) or np.any(positions >= len(full_ids)):
        raise ValueError("anchor position lies outside the corpus")
    full_query_ids, query_vectors, query_offsets = _load_shard(
        args.full_bank / "queries"
    )
    cheap_query_ids, _, _ = _load_shard(args.cheap_bank / "queries")
    if full_query_ids != cheap_query_ids:
        raise ValueError("full/cheap query banks differ")
    query_positions = np.arange(len(full_query_ids), dtype=np.int64)
    anchor_ids = [full_ids[int(index)] for index in positions]
    manifest = write_embedding_bank(
        args.output_bank,
        item_ids=anchor_ids,
        route_embeddings={
            "full-anchor": _views(full_vectors, full_offsets, positions)
        },
        query_ids=full_query_ids,
        query_embeddings=_views(query_vectors, query_offsets, query_positions),
        storage_dtype="float16",
    )
    cheap_manifest = json.loads((args.cheap_bank / "manifest.json").read_text())
    full_manifest = json.loads((args.full_bank / "manifest.json").read_text())
    cheap_bytes = int(cheap_manifest["routes"][args.cheap_route]["vector_bytes"])
    full_bytes = int(full_manifest["routes"][args.full_route]["vector_bytes"])
    anchor_bytes = int(manifest["routes"]["full-anchor"]["vector_bytes"])
    result = {
        "schema_version": 1,
        "architecture": "pool4-cover-plus-full-document-anchors",
        "policy": args.policy,
        "point_index": point_index,
        "anchor_count": len(anchor_ids),
        "corpus_count": len(full_ids),
        "anchor_positions": positions.tolist(),
        "cheap_vector_bytes": cheap_bytes,
        "anchor_vector_bytes": anchor_bytes,
        "combined_vector_bytes": cheap_bytes + anchor_bytes,
        "full_reference_vector_bytes": full_bytes,
        "combined_vector_fraction": (cheap_bytes + anchor_bytes) / full_bytes,
        "bank_manifest": manifest,
        "artifacts": {
            "plan_sha256": _sha256(args.plan),
            "full_manifest_sha256": _sha256(args.full_bank / "manifest.json"),
            "cheap_manifest_sha256": _sha256(args.cheap_bank / "manifest.json"),
        },
    }
    args.output_result.parent.mkdir(parents=True, exist_ok=True)
    args.output_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("anchor_count", "combined_vector_bytes", "combined_vector_fraction")}, indent=2))


if __name__ == "__main__":
    main()
