"""Exact padded-and-masked streamed MaxSim scoring for variable-length banks."""

from __future__ import annotations

from typing import Any

import numpy as np


def streaming_maxsim(
    queries: list[Any],
    documents: list[Any],
    *,
    device: str = "cuda",
    query_chunk: int = 64,
    document_chunk: int = 16,
) -> np.ndarray:
    """Return ColBERT/ColPali MaxSim scores without a query-document Python loop."""
    if not queries or not documents:
        raise ValueError("query and document banks must be non-empty")
    if query_chunk <= 0 or document_chunk <= 0:
        raise ValueError("chunk sizes must be positive")

    import torch
    from torch.nn.utils.rnn import pad_sequence

    dimensions = {int(value.shape[-1]) for value in queries + documents}
    if len(dimensions) != 1:
        raise ValueError("all vectors must share one embedding dimension")
    if any(value.ndim != 2 or len(value) == 0 for value in queries + documents):
        raise ValueError("every bank entry must be a non-empty [tokens, dimension] tensor")

    query_lengths = torch.tensor([len(value) for value in queries], dtype=torch.long)
    document_lengths = torch.tensor([len(value) for value in documents], dtype=torch.long)
    query_bank = pad_sequence([value.float().cpu() for value in queries], batch_first=True)
    document_bank = pad_sequence([value.float().cpu() for value in documents], batch_first=True)
    query_positions = torch.arange(query_bank.shape[1]).unsqueeze(0)
    document_positions = torch.arange(document_bank.shape[1]).unsqueeze(0)
    query_mask = query_positions < query_lengths.unsqueeze(1)
    document_mask = document_positions < document_lengths.unsqueeze(1)
    scores = np.empty((len(queries), len(documents)), dtype=np.float32)

    for query_start in range(0, len(queries), query_chunk):
        query_stop = min(query_start + query_chunk, len(queries))
        query_values = query_bank[query_start:query_stop].to(device)
        valid_query = query_mask[query_start:query_stop].to(device)
        for document_start in range(0, len(documents), document_chunk):
            document_stop = min(document_start + document_chunk, len(documents))
            document_values = document_bank[document_start:document_stop].to(device)
            valid_document = document_mask[document_start:document_stop].to(device)
            similarities = torch.einsum(
                "qlh,dth->qdlt", query_values, document_values
            )
            similarities.masked_fill_(~valid_document[None, :, None, :], -torch.inf)
            block = similarities.amax(-1)
            block.masked_fill_(~valid_query[:, None, :], 0.0)
            scores[query_start:query_stop, document_start:document_stop] = (
                block.sum(-1).cpu().numpy()
            )
        print(
            {"scored_queries": query_stop, "total_queries": len(queries)},
            flush=True,
        )
    return scores
