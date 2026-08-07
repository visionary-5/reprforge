"""Leakage-aware page priorities for progressive materialization baselines."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MaterializationFeatures:
    doc_id: str
    document_id: str
    text_chars: int
    grayscale_entropy: float
    edge_energy: float
    locator_disagreement: float = 0.0
    history_candidate_count: int = 0


def _rank_fusion(rows: Sequence[MaterializationFeatures], fields: Sequence[str]) -> list[str]:
    score = defaultdict(float)
    for field in fields:
        descending = field != "text_chars"
        ranked = sorted(
            rows,
            key=lambda row: (
                -float(getattr(row, field)) if descending else float(getattr(row, field)),
                row.doc_id,
            ),
        )
        for rank, row in enumerate(ranked):
            score[row.doc_id] += 1.0 / (60.0 + rank + 1)
    return sorted(score, key=lambda doc_id: (-score[doc_id], doc_id))


def selection_order(
    rows: Sequence[MaterializationFeatures], *, strategy: str, seed: int
) -> list[str]:
    if not rows:
        raise ValueError("at least one page is required")
    ids = [row.doc_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("page IDs must be unique")
    if strategy == "sha256_random":
        return sorted(
            ids,
            key=lambda doc_id: (
                hashlib.sha256(f"{doc_id}\0{seed}".encode()).digest(),
                doc_id,
            ),
        )
    if strategy == "text_scarcity":
        return [row.doc_id for row in sorted(rows, key=lambda row: (row.text_chars, row.doc_id))]
    if strategy == "visual_complexity":
        return _rank_fusion(rows, ("grayscale_entropy", "edge_energy"))
    if strategy == "cheap_locator_disagreement":
        return [
            row.doc_id
            for row in sorted(rows, key=lambda row: (-row.locator_disagreement, row.doc_id))
        ]
    if strategy == "history_candidate_frequency":
        return [
            row.doc_id
            for row in sorted(rows, key=lambda row: (-row.history_candidate_count, row.doc_id))
        ]
    if strategy == "risk_cover_plus_history_benefit":
        # Three history-benefit admissions for every one coverage admission.
        risk = _rank_fusion(rows, ("text_chars", "grayscale_entropy", "edge_energy"))
        benefit = selection_order(rows, strategy="history_candidate_frequency", seed=seed)
        output: list[str] = []
        seen: set[str] = set()
        cursors = [0, 0]
        schedule = (0, 1, 1, 1)
        while len(output) < len(rows):
            progressed = False
            for source in schedule:
                ranking = risk if source == 0 else benefit
                while cursors[source] < len(ranking) and ranking[cursors[source]] in seen:
                    cursors[source] += 1
                if cursors[source] < len(ranking):
                    doc_id = ranking[cursors[source]]
                    cursors[source] += 1
                    output.append(doc_id)
                    seen.add(doc_id)
                    progressed = True
            if not progressed:
                raise AssertionError("combined selector did not make progress")
        return output
    if strategy == "document_uniform":
        buckets: dict[str, list[MaterializationFeatures]] = defaultdict(list)
        for row in rows:
            buckets[row.document_id].append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda row: (row.text_chars, row.doc_id))
        documents = sorted(buckets)
        output = []
        depth = 0
        while len(output) < len(rows):
            for document in documents:
                if depth < len(buckets[document]):
                    output.append(buckets[document][depth].doc_id)
            depth += 1
        return output
    raise ValueError(f"unsupported strategy: {strategy}")
