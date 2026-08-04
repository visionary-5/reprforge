#!/usr/bin/env python3
"""Compose a calibrated full-corpus runtime from cheap and full-anchor scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reprforge.physical_compression_compiler import (
    calibrated_residual_surface_from_anchors,
)


def _sha256(artifact: Path) -> str:
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(archive: np.lib.npyio.NpzFile, key: str) -> list[str]:
    return [str(value) for value in archive[key]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cheap-runtime", type=Path, required=True)
    parser.add_argument("--anchor-runtime", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    cheap = np.load(args.cheap_runtime, allow_pickle=False)
    anchors = np.load(args.anchor_runtime, allow_pickle=False)
    materialization = json.loads(args.materialization.read_text())
    query_ids = _ids(cheap, "query_ids")
    corpus_ids = _ids(cheap, "corpus_ids")
    if query_ids != _ids(anchors, "query_ids"):
        raise ValueError("cheap and anchor query IDs differ")
    positions = np.asarray(materialization["anchor_positions"], dtype=np.int64)
    expected_anchor_ids = [corpus_ids[int(index)] for index in positions]
    if expected_anchor_ids != _ids(anchors, "corpus_ids"):
        raise ValueError("anchor runtime IDs differ from the materialized plan")
    scores = calibrated_residual_surface_from_anchors(
        cheap["scores"],
        anchors["scores"],
        positions,
        ridge=args.ridge,
    )
    vector_bytes = np.asarray(cheap["vector_bytes"], dtype=np.int64).copy()
    vector_bytes[positions] += np.asarray(anchors["vector_bytes"], dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        query_ids=np.asarray(query_ids),
        corpus_ids=np.asarray(corpus_ids),
        scores=scores.astype(np.float32),
        vector_bytes=vector_bytes,
        encode_ms=np.zeros(len(corpus_ids), dtype=np.float32),
        index_total_ms=np.asarray(0.0, dtype=np.float64),
        model_load_ms=np.asarray(0.0, dtype=np.float64),
    )
    print(
        json.dumps(
            {
                "queries": len(query_ids),
                "corpus": len(corpus_ids),
                "anchors": len(positions),
                "vector_bytes": int(vector_bytes.sum()),
                "cheap_runtime_sha256": _sha256(args.cheap_runtime),
                "anchor_runtime_sha256": _sha256(args.anchor_runtime),
                "materialization_sha256": _sha256(args.materialization),
                "output_sha256": _sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
