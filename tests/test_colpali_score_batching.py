from __future__ import annotations

import pytest

from reprforge.mmdocir_route_runner import ColPaliBackend


def test_colpali_score_batching_matches_scalar_maxsim() -> None:
    torch = pytest.importorskip("torch")
    backend = ColPaliBackend.__new__(ColPaliBackend)
    backend.torch = torch
    backend.device = torch.device("cpu")
    backend.batch_size = 1
    backend.scoring_batch_size = 2
    queries = (
        torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]),
    )
    documents = (
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.0, 0.5]]),
    )
    expected = [
        [
            float((query @ document.T).max(dim=-1).values.sum())
            for document in documents
        ]
        for query in queries
    ]

    observed = backend.score(queries, documents)

    assert torch.tensor(observed) == pytest.approx(torch.tensor(expected))
