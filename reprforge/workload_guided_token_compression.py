"""Homogeneous token compression guided by an unlabeled query workload."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _normalized(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError(f"{name} contains a zero vector")
    return array / norms


def workload_guided_token_selection(
    document_tokens: np.ndarray,
    probes: np.ndarray,
    *,
    budget: int,
    probe_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Select original tokens using weighted fair coverage of query probes.

    Each probe ranks document tokens by cosine similarity. A deterministic
    weighted-fair scheduler repeatedly asks the most under-served probe for
    its best token not already selected. Because the output is a subset of the
    original document tokens, its MaxSim score can never exceed the full-bank
    score for any query.
    """

    documents = _normalized(document_tokens, name="document tokens")
    directions = _normalized(probes, name="workload probes")
    if documents.shape[1] != directions.shape[1]:
        raise ValueError("document tokens and probes must share embedding width")
    if budget <= 0:
        raise ValueError("budget must be positive")
    target = min(int(budget), len(documents))
    if target == len(documents):
        return np.arange(len(documents), dtype=np.int32)
    if probe_weights is None:
        weights = np.ones(len(directions), dtype=np.float64)
    else:
        weights = np.asarray(probe_weights, dtype=np.float64)
        if (
            weights.shape != (len(directions),)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.any(weights > 0)
        ):
            raise ValueError("probe weights must be aligned finite non-negative values")
    weights /= weights.sum()
    similarities = directions @ documents.T
    positions = np.arange(len(documents))
    rankings = np.stack(
        [np.lexsort((positions, -row)) for row in similarities], axis=0
    )
    cursors = np.zeros(len(directions), dtype=np.int64)
    allocations = np.zeros(len(directions), dtype=np.int64)
    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < target:
        # Largest scheduling deficit implements weighted fair queuing without
        # a tuned saliency temperature or per-domain quota.
        expected = (len(selected) + 1) * weights
        probe_order = np.lexsort((np.arange(len(weights)), -(expected - allocations)))
        made_progress = False
        for probe in probe_order:
            probe = int(probe)
            while cursors[probe] < len(documents):
                token = int(rankings[probe, cursors[probe]])
                cursors[probe] += 1
                if token in selected_set:
                    continue
                selected_set.add(token)
                selected.append(token)
                allocations[probe] += 1
                made_progress = True
                break
            if made_progress:
                break
        if not made_progress:
            break
    if len(selected) != target:
        raise RuntimeError("probe scheduler failed to fill the token budget")
    return np.asarray(sorted(selected), dtype=np.int32)


def merge_tokens_to_workload_seeds(
    document_tokens: np.ndarray, selected_positions: Sequence[int]
) -> np.ndarray:
    """Merge every document token into its nearest selected original token."""

    raw = np.asarray(document_tokens, dtype=np.float32)
    normalized = _normalized(raw, name="document tokens")
    selected = np.asarray(selected_positions, dtype=np.int64)
    if (
        selected.ndim != 1
        or not len(selected)
        or len(np.unique(selected)) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= len(raw))
    ):
        raise ValueError("selected positions must be unique document indices")
    assignments = (normalized @ normalized[selected].T).argmax(axis=1)
    output = np.empty((len(selected), raw.shape[1]), dtype=np.float32)
    for cluster in range(len(selected)):
        members = raw[assignments == cluster]
        if not len(members):
            members = raw[selected[cluster] : selected[cluster] + 1]
        center = members.mean(axis=0)
        norm = float(np.linalg.norm(center))
        output[cluster] = center / norm if norm > 0 else raw[selected[cluster]]
    return output
