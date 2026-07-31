from __future__ import annotations

import json

import numpy as np

import reprforge.vidore_local_eval as local_eval
from reprforge.vidore_local_eval import load_local_vidore, write_score_trace


class TracePipeline:
    def export_score_trace(self, query_ids):
        assert list(query_ids) == ["q1", "q2"]
        return {
            "mode": "text",
            "query_ids": np.asarray(["q1", "q2"]),
            "corpus_ids": np.asarray(["d1", "d2", "d3"]),
            "scores": np.asarray(
                [[0.9, 0.3, 0.1], [0.2, 0.4, 0.8]], dtype=np.float32
            ),
            "vector_bytes": np.asarray([8, 12, 16], dtype=np.int64),
            "encode_ms": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "index_total_ms": np.asarray(7.0, dtype=np.float64),
            "model_load_ms": np.asarray(5.0, dtype=np.float64),
        }


def test_load_local_vidore_returns_aligned_official_columns(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "data"
    for part in ("queries", "corpus", "qrels"):
        path = root / part / "test-00000-of-00001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    rows = {
        "queries": [
            {"query_id": "q1", "query": "find d1", "language": "english"},
            {"query_id": "q2", "query": "ignore", "language": "french"},
        ],
        "corpus": [
            {"corpus_id": "d1", "image": b"image", "markdown": "text"}
        ],
        "qrels": [
            {"query_id": "q1", "corpus_id": "d1", "score": 2},
            {"query_id": "q2", "corpus_id": "d1", "score": 1},
        ],
    }
    monkeypatch.setattr(
        local_eval,
        "_read_rows",
        lambda path, columns: rows[path.parent.name],
    )
    monkeypatch.setattr(local_eval, "_decode_image", lambda value: value)

    loaded = load_local_vidore(root, language="english")

    assert loaded[0] == ["q1"]
    assert loaded[1] == ["find d1"]
    assert loaded[2] == ["d1"]
    assert loaded[5] == {"q1": {"d1": 2}}
    assert loaded[7]["selected_queries"] == 1


def test_write_score_trace_separates_runtime_and_oracle_labels(tmp_path) -> None:
    root = tmp_path / "trace"
    manifest = write_score_trace(
        root,
        pipeline=TracePipeline(),  # type: ignore[arg-type]
        query_ids=["q1", "q2"],
        corpus_ids=["d1", "d2", "d3"],
        qrels={"q1": {"d1": 2}, "q2": {"d3": 1}},
        source={"sha256": {"queries": "a", "corpus": "b", "qrels": "c"}},
    )

    with np.load(root / "runtime.npz", allow_pickle=False) as runtime:
        assert runtime["scores"].shape == (2, 3)
        assert float(runtime["index_total_ms"]) == 7.0
        assert "relevance" not in runtime.files
    with np.load(root / "oracle-labels.npz", allow_pickle=False) as labels:
        assert labels["query_positions"].tolist() == [0, 1]
        assert labels["corpus_positions"].tolist() == [0, 2]
        assert labels["relevance"].tolist() == [2, 1]
    assert manifest["labels_are_runtime_visible"] is False
    assert manifest["per_item_encode_ms_sum"] == 6.0
    assert json.loads((root / "manifest.json").read_text())["score_shape"] == [
        2,
        3,
    ]
