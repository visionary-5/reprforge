import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from reprforge.build_mmdocir_slices import build_document_slices
from reprforge.mmdocir_corpus_runner import execute_corpus
from reprforge.mmdocir_route_runner import EncodedBatch
from reprforge.policy_replay import load_replay_data


class FakeBackend:
    def _encode(self, values, width):
        return EncodedBatch(
            tuple(np.ones((1, width), dtype=np.float32) for _ in values),
            tuple(1.0 for _ in values),
        )

    def encode_queries(self, values):
        return self._encode(values, 2)

    def encode_texts(self, values):
        return self._encode(values, 2)

    def encode_images(self, values):
        return self._encode(values, 3)

    def score(self, queries, documents):
        return [[1.0 for _ in documents] for _ in queries]

    def environment(self):
        return {"backend": "fake"}


def test_execute_corpus_merges_document_local_runs(tmp_path: Path) -> None:
    source = tmp_path / "layouts.parquet"
    pq.write_table(
        pa.table(
            {
                "type": ["text", "table"],
                "bbox": [[0.0, 0.0, 1.0, 1.0]] * 2,
                "page_id": [0, 1],
                "text": ["alpha", ""],
                "image_binary": [b"alpha", b"table"],
            }
        ),
        source,
    )
    annotations_path = tmp_path / "annotations.jsonl"
    annotations = [
        {
            "doc_name": "a.pdf",
            "domain": "a",
            "page_indices": [0, 0],
            "layout_indices": [0, 0],
            "questions": [
                {
                    "Q": "alpha?",
                    "page_id": [0],
                    "layout_mapping": [
                        {"page": 0, "bbox": [0.0, 0.0, 1.0, 1.0]}
                    ],
                }
            ],
        },
        {
            "doc_name": "b.pdf",
            "domain": "b",
            "page_indices": [1, 1],
            "layout_indices": [1, 1],
            "questions": [
                {
                    "Q": "table?",
                    "page_id": [0],
                    "layout_mapping": [
                        {"page": 1, "bbox": [0.0, 0.0, 1.0, 1.0]}
                    ],
                }
            ],
        },
    ]
    annotations_path.write_text(
        "".join(json.dumps(row) + "\n" for row in annotations),
        encoding="utf-8",
    )
    build_document_slices(
        layouts_parquet=source,
        annotations_jsonl=annotations_path,
        document_indices=[0, 1],
        output_dir=tmp_path / "slices",
        dataset_revision="dataset-sha",
        source_sha256="source-sha",
        scan_batch_rows=1,
    )

    manifest = execute_corpus(
        backend=FakeBackend(),
        annotations=annotations,
        slice_manifests=sorted((tmp_path / "slices").glob("*.manifest.json")),
        output=tmp_path / "output",
        dataset_revision="dataset-sha",
        model_revision="model-sha",
    )

    assert manifest["document_count"] == 2
    assert manifest["layout_count"] == 2
    assert manifest["query_count"] == 2
    data = load_replay_data(
        tmp_path / "output/items.jsonl",
        tmp_path / "output/queries.jsonl",
        tmp_path / "output/scores.jsonl",
    )
    assert [query.query_id for query in data.queries] == ["query:0", "query:1"]
    baselines = json.loads((tmp_path / "output/baselines.json").read_text())
    assert baselines["fixed-hybrid"]["recall_at_1"] == 1.0

    class FailingBackend:
        def __getattribute__(self, name):
            if name.startswith("encode"):
                raise AssertionError("a complete document was encoded again")
            return super().__getattribute__(name)

    resumed = execute_corpus(
        backend=FailingBackend(),
        annotations=annotations,
        slice_manifests=sorted((tmp_path / "slices").glob("*.manifest.json")),
        output=tmp_path / "output",
        dataset_revision="dataset-sha",
        model_revision="model-sha",
        resume_complete_documents=True,
    )
    assert resumed["document_count"] == 2
    assert all(row["resumed"] for row in resumed["documents"])
