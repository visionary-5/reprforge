"""Artifact loaders for the frozen RBRC v0 protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reprforge.candidate_fusion import _zscore
from reprforge.intervention_utility import _ndcg_row
from reprforge.rbrc_v0 import DomainSurface


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _surface_domain(
    *,
    name: str,
    query_ids: np.ndarray,
    corpus_ids: np.ndarray,
    locator_scores: np.ndarray,
    reranker_scores: np.ndarray,
    qrels: np.ndarray,
    depths: Sequence[int],
    input_sha256: dict[str, str],
) -> DomainSurface:
    if locator_scores.shape != reranker_scores.shape or qrels.shape != locator_scores.shape:
        raise ValueError(f"{name}: score/qrel surfaces are unaligned")
    if locator_scores.shape != (len(query_ids), len(corpus_ids)):
        raise ValueError(f"{name}: IDs do not match score shape")
    positions = np.arange(len(corpus_ids))
    ranking: dict[str, list[str]] = {}
    quality: dict[str, dict[int, float]] = {}
    for query_position, raw_query_id in enumerate(query_ids):
        query_id = str(raw_query_id)
        order = np.lexsort((positions, -locator_scores[query_position]))[
            : max(depths)
        ]
        ranking[query_id] = [str(corpus_ids[index]) for index in order]
        quality[query_id] = {}
        for depth in depths:
            candidates = order[:depth]
            fused = _zscore(locator_scores[query_position, candidates]) + _zscore(
                reranker_scores[query_position, candidates]
            )
            scores = np.full(len(corpus_ids), -np.inf, dtype=np.float64)
            scores[candidates] = fused
            quality[query_id][int(depth)] = _ndcg_row(
                scores,
                qrels[query_position],
                np.asarray(corpus_ids),
                cutoff=10,
            )
    domain = DomainSurface(
        name=name,
        corpus_pages=len(corpus_ids),
        query_ids=tuple(str(value) for value in query_ids),
        ranking=ranking,
        quality=quality,
        input_sha256=input_sha256,
    )
    domain.validate(depths)
    return domain


def load_bm25_colpali_domain(
    name: str, root: Path, depths: Sequence[int]
) -> DomainSurface:
    text_path = root / "bm25" / "runtime.npz"
    visual_path = root / "visual" / "runtime.npz"
    labels_path = root / "bm25" / "oracle-labels.npz"
    with np.load(text_path, allow_pickle=False) as source:
        text = {key: source[key] for key in source.files}
    with np.load(visual_path, allow_pickle=False) as source:
        visual = {key: source[key] for key in source.files}
    with np.load(labels_path, allow_pickle=False) as source:
        labels = {key: source[key] for key in source.files}
    if not np.array_equal(text["query_ids"], visual["query_ids"]):
        raise ValueError(f"{name}: text and visual query IDs differ")
    if not np.array_equal(text["corpus_ids"], visual["corpus_ids"]):
        raise ValueError(f"{name}: text and visual corpus IDs differ")
    qrels = np.zeros(np.asarray(text["scores"]).shape, dtype=np.int16)
    qrels[labels["query_positions"], labels["corpus_positions"]] = labels[
        "relevance"
    ]
    return _surface_domain(
        name=name,
        query_ids=text["query_ids"],
        corpus_ids=text["corpus_ids"],
        locator_scores=np.asarray(text["scores"], dtype=np.float64),
        reranker_scores=np.asarray(visual["scores"], dtype=np.float64),
        qrels=qrels,
        depths=depths,
        input_sha256={
            "locator_runtime": sha256(text_path),
            "reranker_runtime": sha256(visual_path),
            "qrels": sha256(labels_path),
        },
    )


def load_irpapers_domain(
    name: str, surface_path: Path, query_csv: Path, depths: Sequence[int]
) -> DomainSurface:
    with np.load(surface_path, allow_pickle=False) as source:
        surface = {key: source[key] for key in source.files}
    with query_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    query_ids = np.asarray(surface["query_ids"])
    corpus_ids = np.asarray(surface["corpus_ids"])
    if len(rows) != len(query_ids):
        raise ValueError("IRPAPERS query CSV and score surface have different lengths")
    corpus_position = {str(value): index for index, value in enumerate(corpus_ids)}
    qrels = np.zeros((len(query_ids), len(corpus_ids)), dtype=np.int16)
    for query_position, row in enumerate(rows):
        gold_id = str(row["dataset_id"])
        if gold_id not in corpus_position:
            raise ValueError(f"IRPAPERS gold page missing from corpus: {gold_id}")
        qrels[query_position, corpus_position[gold_id]] = 1
    return _surface_domain(
        name=name,
        query_ids=query_ids,
        corpus_ids=corpus_ids,
        locator_scores=np.asarray(surface["bm25_scores"], dtype=np.float64),
        reranker_scores=np.asarray(surface["visual_scores"], dtype=np.float64),
        qrels=qrels,
        depths=depths,
        input_sha256={
            "score_surface": sha256(surface_path),
            "queries": sha256(query_csv),
        },
    )


def _query_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def load_omni_domain(
    name: str, failure_path: Path, ranking_path: Path, depths: Sequence[int]
) -> DomainSurface:
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    rows = failure.get("per_query", [])
    corpus_pages = int(failure.get("analysis_scope", {}).get("corpus_pages", 0))
    ranking: dict[str, list[str]] = {}
    with ranking_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(
                    f"expected 3 tab fields at {ranking_path}:{line_number}"
                )
            query_id, document_id, _ = fields
            ranking.setdefault(query_id, []).append(document_id)
    quality = {
        str(row["query_id"]): {
            int(depth): float(row["ndcg_at_10"][f"cascade{depth}"])
            for depth in depths
        }
        for row in rows
    }
    domain = DomainSurface(
        name=name,
        corpus_pages=corpus_pages,
        query_ids=tuple(sorted(ranking, key=_query_sort_key)),
        ranking=ranking,
        quality=quality,
        input_sha256={
            "failure_analysis": sha256(failure_path),
            "hpool_ranking": sha256(ranking_path),
        },
    )
    domain.validate(depths)
    return domain

