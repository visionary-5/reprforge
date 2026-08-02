"""Small deterministic BM25 core shared by online and trace execution."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np


Bm25State = tuple[dict[str, list[tuple[int, int]]], np.ndarray, float, int]


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def build_index(
    documents: Sequence[str],
) -> tuple[Bm25State, np.ndarray, int]:
    tokenized = [tokenize(value) for value in documents]
    lengths = np.asarray([len(value) for value in tokenized], dtype=np.float64)
    average_length = max(float(lengths.mean()), 1.0)
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    posting_bytes = np.zeros(len(documents), dtype=np.int64)
    for position, tokens in enumerate(tokenized):
        for term, frequency in Counter(tokens).items():
            postings[term].append((position, frequency))
            # Logical accounting: uint32 doc id + uint32 term frequency.
            posting_bytes[position] += 8
    vocabulary_bytes = sum(len(term.encode("utf-8")) for term in postings)
    return (
        (dict(postings), lengths, average_length, len(documents)),
        posting_bytes,
        vocabulary_bytes,
    )


def score_queries(
    state: Bm25State,
    queries: Sequence[str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> np.ndarray:
    postings, lengths, average_length, document_count = state
    scores = np.zeros((len(queries), document_count), dtype=np.float32)
    for query_position, query in enumerate(queries):
        for term, query_frequency in Counter(tokenize(query)).items():
            values = postings.get(term)
            if not values:
                continue
            document_frequency = len(values)
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for document_position, frequency in values:
                denominator = frequency + k1 * (
                    1.0 - b + b * lengths[document_position] / average_length
                )
                scores[query_position, document_position] += float(
                    query_frequency
                    * inverse_document_frequency
                    * frequency
                    * (k1 + 1.0)
                    / denominator
                )
    return scores


def scores(
    documents: Sequence[str],
    queries: Sequence[str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, int]:
    state, posting_bytes, vocabulary_bytes = build_index(documents)
    return (
        score_queries(state, queries, k1=k1, b=b),
        posting_bytes,
        vocabulary_bytes,
    )
