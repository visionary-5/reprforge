#!/usr/bin/env python3
"""Replay static mixed-representation witnesses from frozen ViDoRe traces.

This module deliberately separates a feasible *diagnostic witness* from a
deployable policy.  The strongest selector uses qrels and complete text/visual
rank outcomes.  It proves that a small resident visual set exists; it cannot
be used online or reported as a learned policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_COUNTS = (0, 25, 50, 75, 100, 111, 150, 222, 333, 555, 888, 1110)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenTrace:
    root: Path
    manifest: dict[str, Any]
    query_ids: np.ndarray
    corpus_ids: np.ndarray
    scores: np.ndarray
    vector_bytes: np.ndarray
    encode_ms: np.ndarray
    index_total_ms: float
    label_query: np.ndarray
    label_corpus: np.ndarray
    relevance: np.ndarray


def load_trace(root: Path) -> FrozenTrace:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_path = root / manifest["runtime_file"]
    labels_path = root / manifest["oracle_labels_file"]
    if _sha256(runtime_path) != manifest["runtime_sha256"]:
        raise ValueError(f"runtime trace digest mismatch: {runtime_path}")
    if _sha256(labels_path) != manifest["oracle_labels_sha256"]:
        raise ValueError(f"oracle-label digest mismatch: {labels_path}")
    with np.load(runtime_path, allow_pickle=False) as runtime:
        values = {key: runtime[key] for key in runtime.files}
    with np.load(labels_path, allow_pickle=False) as labels:
        label_values = {key: labels[key] for key in labels.files}
    scores = np.asarray(values["scores"], dtype=np.float64)
    if list(scores.shape) != manifest["score_shape"]:
        raise ValueError("runtime score shape differs from manifest")
    return FrozenTrace(
        root=root,
        manifest=manifest,
        query_ids=np.asarray(values["query_ids"]),
        corpus_ids=np.asarray(values["corpus_ids"]),
        scores=scores,
        vector_bytes=np.asarray(values["vector_bytes"], dtype=np.int64),
        encode_ms=np.asarray(values["encode_ms"], dtype=np.float64),
        index_total_ms=float(values["index_total_ms"]),
        label_query=np.asarray(label_values["query_positions"], dtype=np.int32),
        label_corpus=np.asarray(
            label_values["corpus_positions"], dtype=np.int32
        ),
        relevance=np.asarray(label_values["relevance"], dtype=np.int16),
    )


def validate_pair(text: FrozenTrace, visual: FrozenTrace) -> np.ndarray:
    if text.manifest["mode"] != "text" or visual.manifest["mode"] != "visual":
        raise ValueError("expected one text trace and one visual trace")
    if not np.array_equal(text.query_ids, visual.query_ids):
        raise ValueError("text and visual query identifiers differ")
    if not np.array_equal(text.corpus_ids, visual.corpus_ids):
        raise ValueError("text and visual corpus identifiers differ")
    for field in ("label_query", "label_corpus", "relevance"):
        if not np.array_equal(getattr(text, field), getattr(visual, field)):
            raise ValueError(f"text and visual {field} arrays differ")
    if text.manifest["source_sha256"] != visual.manifest["source_sha256"]:
        raise ValueError("text and visual source revisions differ")
    qrels = np.zeros(text.scores.shape, dtype=np.int16)
    qrels[text.label_query, text.label_corpus] = text.relevance
    if np.any(qrels.max(axis=1) == 0):
        raise ValueError("at least one query has no relevance label")
    return qrels


def rank_order(scores: np.ndarray, corpus_ids: np.ndarray) -> np.ndarray:
    """Match ReprForge's score-descending, identifier-ascending ordering."""

    return np.stack(
        [np.lexsort((corpus_ids, -row)) for row in scores],
        axis=0,
    )


def mean_ndcg(
    scores: np.ndarray,
    qrels: np.ndarray,
    corpus_ids: np.ndarray,
    *,
    cutoff: int = 10,
) -> float:
    if scores.shape != qrels.shape:
        raise ValueError("score and qrel matrices differ in shape")
    order = rank_order(scores, corpus_ids)[:, :cutoff]
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
    gains = np.take_along_axis(qrels, order, axis=1)
    ideal = np.sort(qrels, axis=1)[:, -1 : -cutoff - 1 : -1]
    denominator = (ideal * discounts).sum(axis=1)
    if np.any(denominator == 0):
        raise ValueError("nDCG is undefined for a query without relevant pages")
    return float(np.mean((gains * discounts).sum(axis=1) / denominator))


def _discount(rank: np.ndarray, cutoff: int) -> np.ndarray:
    result = np.zeros(rank.shape, dtype=np.float64)
    visible = rank < cutoff
    result[visible] = 1.0 / np.log2(rank[visible] + 2.0)
    return result


def selector_orders(
    text: FrozenTrace,
    visual: FrozenTrace,
    *,
    cutoff: int = 10,
    touch_k: int = 20,
) -> dict[str, tuple[np.ndarray, bool, str]]:
    corpus_count = text.scores.shape[1]
    qrel_frequency = np.bincount(
        text.label_corpus,
        weights=text.relevance,
        minlength=corpus_count,
    )
    text_order = rank_order(text.scores, text.corpus_ids)
    touch_frequency = np.bincount(
        text_order[:, :touch_k].ravel(), minlength=corpus_count
    )

    text_rank = np.empty_like(text_order, dtype=np.int32)
    visual_order = rank_order(visual.scores, visual.corpus_ids)
    visual_rank = np.empty_like(visual_order, dtype=np.int32)
    positions = np.arange(corpus_count, dtype=np.int32)
    text_rank[np.arange(len(text_order))[:, None], text_order] = positions
    visual_rank[np.arange(len(visual_order))[:, None], visual_order] = positions
    labelled_text_rank = text_rank[text.label_query, text.label_corpus]
    labelled_visual_rank = visual_rank[text.label_query, text.label_corpus]
    relevant_rank_benefit = np.zeros(corpus_count, dtype=np.float64)
    benefit = text.relevance * np.maximum(
        0.0,
        _discount(labelled_visual_rank, cutoff)
        - _discount(labelled_text_rank, cutoff),
    )
    np.add.at(relevant_rank_benefit, text.label_corpus, benefit)

    return {
        "text_top20_frequency": (
            np.argsort(-touch_frequency, kind="stable"),
            False,
            "future-aware full-stream frequency; online LFU must use past only",
        ),
        "qrel_frequency": (
            np.argsort(-qrel_frequency, kind="stable"),
            False,
            "oracle-only label frequency; ignores ranking interventions",
        ),
        "positive_relevant_rank_sensitivity": (
            np.argsort(-relevant_rank_benefit, kind="stable"),
            False,
            "oracle-only qrels plus full text/visual rank interventions",
        ),
    }


def analyze(
    text: FrozenTrace,
    visual: FrozenTrace,
    *,
    counts: Sequence[int] = DEFAULT_COUNTS,
    cutoff: int = 10,
) -> dict[str, Any]:
    qrels = validate_pair(text, visual)
    corpus_count = text.scores.shape[1]
    normalized_counts = sorted(
        {min(corpus_count, max(0, int(value))) for value in counts}
    )
    text_metric = mean_ndcg(text.scores, qrels, text.corpus_ids, cutoff=cutoff)
    visual_metric = mean_ndcg(
        visual.scores, qrels, visual.corpus_ids, cutoff=cutoff
    )
    target_metric = text_metric + 0.95 * (visual_metric - text_metric)
    full_visual_bytes = int(visual.vector_bytes.sum())
    base_text_bytes = int(text.vector_bytes.sum())
    selectors: dict[str, Any] = {}
    for name, (order, runtime_visible, description) in selector_orders(
        text, visual, cutoff=cutoff
    ).items():
        curve = []
        for count in normalized_counts:
            selected = order[:count]
            mixed = text.scores.copy()
            mixed[:, selected] = visual.scores[:, selected]
            metric = mean_ndcg(mixed, qrels, text.corpus_ids, cutoff=cutoff)
            resident_visual_bytes = int(visual.vector_bytes[selected].sum())
            curve.append(
                {
                    "resident_count": count,
                    "resident_fraction": count / corpus_count,
                    f"ndcg@{cutoff}": metric,
                    "gain_retained": (
                        (metric - text_metric) / (visual_metric - text_metric)
                    ),
                    "base_plus_visual_bytes": (
                        base_text_bytes + resident_visual_bytes
                    ),
                    "resident_visual_bytes": resident_visual_bytes,
                    "estimated_batched_build_ms": (
                        text.index_total_ms + float(visual.encode_ms[selected].sum())
                    ),
                    "passes_quality_target": metric >= target_metric,
                    "passes_space_target": (
                        base_text_bytes + resident_visual_bytes
                        < full_visual_bytes
                    ),
                }
            )
        eligible = [
            point
            for point in curve
            if point["resident_fraction"] <= 0.30
            and point["passes_quality_target"]
            and point["passes_space_target"]
        ]
        best = max(
            (point for point in curve if point["resident_fraction"] <= 0.30),
            key=lambda point: point[f"ndcg@{cutoff}"],
        )
        selectors[name] = {
            "runtime_deployable_as_written": runtime_visible,
            "description": description,
            "curve": curve,
            "best_at_or_below_30_percent": best,
            "first_gate_witness": eligible[0] if eligible else None,
        }

    # Count directly rather than infer it from the sorted order.  Recompute the
    # sparse benefit vector to keep the public output explicit and testable.
    text_order = rank_order(text.scores, text.corpus_ids)
    visual_order = rank_order(visual.scores, visual.corpus_ids)
    text_rank = np.empty_like(text_order, dtype=np.int32)
    visual_rank = np.empty_like(visual_order, dtype=np.int32)
    positions = np.arange(corpus_count, dtype=np.int32)
    text_rank[np.arange(len(text_order))[:, None], text_order] = positions
    visual_rank[np.arange(len(visual_order))[:, None], visual_order] = positions
    sparse_benefit = text.relevance * np.maximum(
        0.0,
        _discount(text_rank[text.label_query, text.label_corpus], cutoff) * -1
        + _discount(visual_rank[text.label_query, text.label_corpus], cutoff),
    )
    page_benefit = np.zeros(corpus_count, dtype=np.float64)
    np.add.at(page_benefit, text.label_corpus, sparse_benefit)
    positive_sensitivity_pages = int(np.count_nonzero(page_benefit > 0))

    return {
        "schema_version": 1,
        "semantics": {
            "fusion": "raw score replacement for resident pages",
            "base_representation_remains_resident": True,
            "ndcg_gain": "linear qrel gain matching pytrec_eval ndcg_cut",
            "cost_estimate": (
                "full-corpus batched per-item encode timings; subset batch "
                "edge effects are not measured"
            ),
        },
        "source": {
            "text_runtime_sha256": text.manifest["runtime_sha256"],
            "visual_runtime_sha256": visual.manifest["runtime_sha256"],
            "oracle_labels_sha256": text.manifest["oracle_labels_sha256"],
            "official_upstream_commit": text.manifest[
                "official_upstream_commit"
            ],
            "data_sha256": text.manifest["source_sha256"],
        },
        "workload": {
            "queries": int(text.scores.shape[0]),
            "corpus": corpus_count,
            "labelled_pairs": int(len(text.relevance)),
            "unique_relevant_pages": int(np.unique(text.label_corpus).size),
            "positive_relevant_rank_sensitivity_pages": (
                positive_sensitivity_pages
            ),
        },
        "baselines": {
            f"text_ndcg@{cutoff}": text_metric,
            f"visual_ndcg@{cutoff}": visual_metric,
            f"target_ndcg@{cutoff}": target_metric,
            "text_index_bytes": base_text_bytes,
            "visual_index_bytes": full_visual_bytes,
            "text_index_ms": text.index_total_ms,
            "visual_index_ms": visual.index_total_ms,
        },
        "selectors": selectors,
        "verdict": (
            "static diagnostic headroom exists; no deployable admission "
            "policy is established"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-trace", type=Path, required=True)
    parser.add_argument("--visual-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--counts", type=int, nargs="*", default=DEFAULT_COUNTS)
    args = parser.parse_args()
    result = analyze(
        load_trace(args.text_trace),
        load_trace(args.visual_trace),
        counts=args.counts,
        cutoff=args.cutoff,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
