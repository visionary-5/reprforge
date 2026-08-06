import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.rerank_omni_sharded_candidates import run_sharded_cascade


def _shard(path: Path, doc_ids: list[str], vectors: list[list[float]]) -> Path:
    path.mkdir()
    tensor = torch.tensor(vectors, dtype=torch.float32).reshape(len(doc_ids), 1, 2)
    torch.save(tensor, path / "index.pt")
    torch.save(torch.ones((len(doc_ids), 1), dtype=torch.bool), path / "masks.pt")
    with (path / "metadata.pkl").open("wb") as handle:
        pickle.dump({"doc_ids": doc_ids}, handle)
    return path


def test_sharded_candidate_reranking_reads_the_owning_shard(tmp_path: Path) -> None:
    left = _shard(tmp_path / "left", ["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    right = _shard(tmp_path / "right", ["c", "d"], [[2.0, 0.0], [-1.0, 0.0]])
    candidates = tmp_path / "candidates.tsv"
    candidates.write_text("q\ta\t4\nq\tb\t3\nq\tc\t2\nq\td\t1\n", encoding="utf-8")
    embeddings = tmp_path / "queries.pkl"
    masks = tmp_path / "masks.pkl"
    with embeddings.open("wb") as handle:
        pickle.dump((np.array([[[1.0, 0.0]]], dtype=np.float32), ["q"]), handle)
    with masks.open("wb") as handle:
        pickle.dump((np.array([[True]]), ["q"]), handle)
    output = tmp_path / "output"
    run_sharded_cascade(
        [left, right],
        embeddings,
        masks,
        candidates,
        output,
        candidate_depth=4,
        rerank_depths=(2, 4),
        score_chunk_size=1,
        device="cpu",
    )
    rows = (output / "cascade-top4.ranking.tsv").read_text().splitlines()
    assert [row.split("\t")[1] for row in rows] == ["c", "a", "b", "d"]
