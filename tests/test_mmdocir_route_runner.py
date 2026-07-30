import json
from pathlib import Path

import numpy as np
import pytest

from reprforge.mmdocir_route_runner import ColPaliBackend, EncodedBatch, execute_document
from reprforge.policy_replay import load_replay_data


class FakeBackend:
    def _encode(self, values, width):
        embeddings = tuple(
            np.full((index + 1, width), index + 1, dtype=np.float32)
            for index, _ in enumerate(values)
        )
        return EncodedBatch(
            embeddings=embeddings,
            encode_ms=tuple(float(index + 1) for index, _ in enumerate(values)),
        )

    def encode_queries(self, queries):
        return self._encode(queries, 2)

    def encode_texts(self, texts):
        return self._encode(texts, 2)

    def encode_images(self, images):
        return self._encode(images, 3)

    def score(self, queries, documents):
        return [
            [float((query_index + 1) * (document_index + 1))
             for document_index in range(len(documents))]
            for query_index in range(len(queries))
        ]

    def environment(self):
        return {"backend": "fake"}

    def derive_image_routes(self, images):
        return {
            "image-pool-2": EncodedBatch(
                embeddings=tuple(value[:1] for value in images.embeddings),
                encode_ms=tuple(value + 0.5 for value in images.encode_ms),
            )
        }

    def construction_features(self, images):
        return [
            {"grayscale_entropy": float(index), "feature_extract_ms": 0.1}
            for index, _ in enumerate(images)
        ]

    def representation_features(self):
        return {
            "image-pool-2": [
                {"cosine_cover_loss_max": float(index + 1) / 10.0}
                for index in range(2)
            ]
        }


class FakeProcessor:
    image_seq_length = 2

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {
            "input_ids": np.asarray([[10, 11, 12, 13]]),
            "attention_mask": np.asarray([[1, 1, 1, 1]]),
            "pixel_values": np.zeros((1, 1), dtype=np.float32),
        }


def test_text_batch_matches_official_mock_image_strip_contract() -> None:
    backend = object.__new__(ColPaliBackend)
    backend.processor = FakeProcessor()
    backend.mock_image = object()

    batch = backend._text_batch(["evidence"], "Passage: ", max_length=600)

    assert "pixel_values" not in batch
    assert batch["input_ids"].tolist() == [[12, 13]]
    assert batch["attention_mask"].tolist() == [[1, 1]]
    assert backend.processor.kwargs["text"] == [
        "Passage: evidence" + "<pad>" * 10
    ]
    assert backend.processor.kwargs["images"] == [backend.mock_image]
    assert backend.processor.kwargs["max_length"] == 602


def test_maxsim_scoring_masks_variable_document_padding() -> None:
    torch = pytest.importorskip("torch")
    backend = object.__new__(ColPaliBackend)
    backend.torch = torch
    backend.device = torch.device("cpu")
    backend.batch_size = 2
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    documents = (
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.5, 0.0], [0.0, 0.25]]),
    )

    scores = backend.score((query,), documents)

    assert np.allclose(scores, [[1.0, 0.75]])


def test_execute_document_exports_replayable_costed_routes(tmp_path: Path) -> None:
    annotations = [
        {
            "doc_name": "fixture.pdf",
            "domain": "Academic paper",
            "page_indices": [0, 0],
            "layout_indices": [10, 11],
            "questions": [
                {
                    "Q": "Where is the answer?",
                    "page_id": [0],
                    "layout_mapping": [
                        {"page": 0, "bbox": [0.0, 0.0, 1.0, 1.0]}
                    ],
                }
            ],
        }
    ]
    layout_rows = [
        {
            "type": "text",
            "page_id": 0,
            "bbox": [0.0, 0.0, 0.5, 1.0],
            "text": "left",
            "image_binary": b"left-image",
        },
        {
            "type": "table",
            "page_id": 0,
            "bbox": [0.5, 0.0, 1.0, 1.0],
            "text": "",
            "image_binary": b"right-image",
        },
    ]
    manifest = execute_document(
        backend=FakeBackend(),
        annotations=annotations,
        document_index=0,
        layout_rows=layout_rows,
        output=tmp_path,
        dataset_revision="dataset-sha",
        model_revision="model-sha",
        persist_embeddings=True,
    )
    assert manifest["fixed_hybrid_direct_replay_equal"] is True
    assert manifest["layouts"] == 2
    data = load_replay_data(
        tmp_path / "items.jsonl",
        tmp_path / "queries.jsonl",
        tmp_path / "scores.jsonl",
    )
    assert data.items[0].route_costs["text"].index_bytes == 8
    assert data.items[0].route_costs["image"].index_bytes == 12
    baselines = json.loads((tmp_path / "baselines.json").read_text())
    assert set(baselines) == {
        "all-image",
        "all-image-pool-2",
        "all-text",
        "fixed-hybrid",
        "fixed-hybrid-image-pool-2",
    }
    assert data.items[0].route_costs["image-pool-2"].index_bytes == 12
    assert manifest["routes"] == ["image", "image-pool-2", "text"]
    assert manifest["embeddings_persisted"] is True
    assert manifest["embedding_bank"]["item_count"] == 2
    assert manifest["embedding_bank"]["query_count"] == 1
    bank_manifest = json.loads(
        (tmp_path / "embedding-bank" / "manifest.json").read_text()
    )
    assert set(bank_manifest["routes"]) == {"image", "image-pool-2", "text"}
    item_row = json.loads((tmp_path / "items.jsonl").read_text().splitlines()[0])
    assert item_row["construction_features"]["grayscale_entropy"] == 0.0
    assert item_row["route_features"]["text"]["vector_count"] == 1
    assert item_row["route_features"]["image"]["embedding_dimension"] == 3
    assert (
        item_row["route_features"]["image-pool-2"]["cosine_cover_loss_max"]
        == 0.1
    )
    assert json.loads((tmp_path / "environment.json").read_text())["backend"] == "fake"
