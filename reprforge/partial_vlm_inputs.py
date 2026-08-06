"""Load existing complete text/visual score surfaces for a what-if audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from reprforge.partial_vlm_materialization import ScoreSurface


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def load_vidore_surface(name: str, root: Path) -> ScoreSurface:
    text_path = root / "bm25" / "runtime.npz"
    visual_path = root / "visual" / "runtime.npz"
    labels_path = root / "bm25" / "oracle-labels.npz"
    text = _load_npz(text_path)
    visual = _load_npz(visual_path)
    labels = _load_npz(labels_path)
    if not np.array_equal(text["query_ids"], visual["query_ids"]):
        raise ValueError(f"{name}: text and visual query IDs differ")
    if not np.array_equal(text["corpus_ids"], visual["corpus_ids"]):
        raise ValueError(f"{name}: text and visual corpus IDs differ")
    qrels = np.zeros(np.asarray(text["scores"]).shape, dtype=np.int16)
    qrels[labels["query_positions"], labels["corpus_positions"]] = labels[
        "relevance"
    ]
    return ScoreSurface(
        name=name,
        query_ids=np.asarray(text["query_ids"]),
        corpus_ids=np.asarray(text["corpus_ids"]),
        text_scores=np.asarray(text["scores"], dtype=np.float64),
        visual_scores=np.asarray(visual["scores"], dtype=np.float64),
        qrels=qrels,
        text_bytes=np.asarray(text["vector_bytes"], dtype=np.float64),
        visual_bytes=np.asarray(visual["vector_bytes"], dtype=np.float64),
        visual_encode_ms=np.asarray(visual["encode_ms"], dtype=np.float64),
        input_sha256={
            "text_runtime": sha256(text_path),
            "visual_runtime": sha256(visual_path),
            "qrels": sha256(labels_path),
        },
    )


def load_irpapers_surface(
    name: str, surface_path: Path, query_csv: Path, run_manifest: Path
) -> ScoreSurface:
    surface = _load_npz(surface_path)
    with query_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    query_ids = np.asarray(surface["query_ids"])
    corpus_ids = np.asarray(surface["corpus_ids"])
    if len(rows) != len(query_ids):
        raise ValueError("IRPAPERS query metadata and score surface differ")
    corpus_position = {str(value): index for index, value in enumerate(corpus_ids)}
    qrels = np.zeros((len(query_ids), len(corpus_ids)), dtype=np.int16)
    for query_position, row in enumerate(rows):
        gold_id = str(row["dataset_id"])
        if gold_id not in corpus_position:
            raise ValueError(f"IRPAPERS gold page missing: {gold_id}")
        qrels[query_position, corpus_position[gold_id]] = 1
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    full_cost = manifest["runs"]["full_visual_colpali_v1_1"]["cost"]
    average_bytes = float(full_cost["index_vector_bytes"]) / len(corpus_ids)
    average_encode_ms = float(full_cost["measured_index_ms_inside_pipeline"]) / len(
        corpus_ids
    )
    return ScoreSurface(
        name=name,
        query_ids=query_ids,
        corpus_ids=corpus_ids,
        text_scores=np.asarray(surface["bm25_scores"], dtype=np.float64),
        visual_scores=np.asarray(surface["visual_scores"], dtype=np.float64),
        qrels=qrels,
        text_bytes=np.ones(len(corpus_ids), dtype=np.float64),
        visual_bytes=np.full(len(corpus_ids), average_bytes, dtype=np.float64),
        visual_encode_ms=np.full(len(corpus_ids), average_encode_ms, dtype=np.float64),
        input_sha256={
            "score_surface": sha256(surface_path),
            "queries": sha256(query_csv),
            "run_manifest": sha256(run_manifest),
        },
    )

