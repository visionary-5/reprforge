"""Qrel-free page selection for physical partial visual indexes."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PageRiskFeatures:
    doc_id: str
    text_chars: int
    grayscale_entropy: float
    edge_energy: float
    nonwhite_fraction: float


def selection_order(
    features: Sequence[PageRiskFeatures], *, strategy: str, seed: int
) -> list[str]:
    if not features:
        raise ValueError("at least one page is required")
    ids = [row.doc_id for row in features]
    if len(set(ids)) != len(ids):
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
        return [
            row.doc_id
            for row in sorted(features, key=lambda row: (row.text_chars, row.doc_id))
        ]
    if strategy != "risk_cover_round_robin":
        raise ValueError(f"unsupported strategy: {strategy}")
    rankings = (
        sorted(features, key=lambda row: (row.text_chars, row.doc_id)),
        sorted(features, key=lambda row: (-row.grayscale_entropy, row.doc_id)),
        sorted(features, key=lambda row: (-row.edge_energy, row.doc_id)),
    )
    output: list[str] = []
    seen: set[str] = set()
    cursors = [0] * len(rankings)
    while len(output) < len(features):
        added = False
        for ranking_index, ranking in enumerate(rankings):
            while (
                cursors[ranking_index] < len(ranking)
                and ranking[cursors[ranking_index]].doc_id in seen
            ):
                cursors[ranking_index] += 1
            if cursors[ranking_index] < len(ranking):
                doc_id = ranking[cursors[ranking_index]].doc_id
                cursors[ranking_index] += 1
                seen.add(doc_id)
                output.append(doc_id)
                added = True
        if not added:
            raise AssertionError("risk-cover selector did not make progress")
    return output


def budget_count(item_count: int, fraction: float) -> int:
    if item_count <= 0:
        raise ValueError("item_count must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return min(item_count, max(1, math.ceil(item_count * fraction)))
