"""Workload-conditioned token witnesses for late-interaction indexes."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def compile_token_witnesses(
    winner_indices: np.ndarray,
    *,
    fit_queries: Sequence[int],
    document_token_counts: Sequence[int],
    minimum_win_count: int = 1,
    minimum_tokens: int = 8,
    competitive_pairs: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Compile one variable-size token set per document.

    ``winner_indices[q, d]`` contains the document-token positions that attain
    MaxSim for every valid token of query ``q`` against document ``d``.  A
    token is retained when it wins often enough on the fit workload.  The
    deterministic minimum-size fallback keeps the most frequent witnesses.
    """

    winners = np.asarray(winner_indices)
    counts = np.asarray(document_token_counts, dtype=np.int64)
    fit = np.asarray(fit_queries, dtype=np.int64)
    if winners.ndim != 3 or winners.shape[1] != len(counts):
        raise ValueError("winner tensor and document token counts are not aligned")
    if len(fit) == 0 or np.any(fit < 0) or np.any(fit >= winners.shape[0]):
        raise ValueError("fit query indices are empty or out of range")
    if minimum_win_count <= 0 or minimum_tokens < 0:
        raise ValueError("win threshold must be positive and minimum tokens non-negative")
    pairs = None
    if competitive_pairs is not None:
        pairs = np.asarray(competitive_pairs, dtype=bool)
        if pairs.shape != winners.shape[:2]:
            raise ValueError("competitive pair mask must be query-by-document aligned")
    plans = []
    for document, token_count in enumerate(counts):
        if token_count <= 0:
            raise ValueError("documents must contain at least one token")
        all_values = winners[fit, document].ravel()
        all_values = all_values[(all_values >= 0) & (all_values < token_count)]
        active_fit = fit if pairs is None else fit[pairs[fit, document]]
        values = winners[active_fit, document].ravel()
        values = values[(values >= 0) & (values < token_count)]
        frequency = np.bincount(values, minlength=int(token_count))
        fallback_frequency = np.bincount(
            all_values, minlength=int(token_count)
        )
        selected = np.flatnonzero(frequency >= minimum_win_count)
        target = min(int(token_count), minimum_tokens)
        if len(selected) < target:
            order = np.lexsort((np.arange(token_count), -fallback_frequency))
            selected = np.unique(np.concatenate([selected, order[:target]]))
        plans.append(np.asarray(sorted(selected.tolist()), dtype=np.int32))
    return tuple(plans)


def matched_random_witnesses(
    witnesses: Sequence[Sequence[int]],
    *,
    document_token_counts: Sequence[int],
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Draw a deterministic per-document random baseline at matched sizes."""

    if len(witnesses) != len(document_token_counts):
        raise ValueError("witnesses and token counts are not aligned")
    generator = np.random.default_rng(seed)
    output = []
    for values, token_count in zip(
        witnesses, document_token_counts, strict=True
    ):
        size = len(values)
        if not 0 <= size <= token_count:
            raise ValueError("matched witness size is invalid")
        output.append(
            np.asarray(
                sorted(generator.choice(token_count, size=size, replace=False)),
                dtype=np.int32,
            )
        )
    return tuple(output)
