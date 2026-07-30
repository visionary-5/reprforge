import json
from pathlib import Path

from reprforge.merge_replay_banks import merge_replay_banks
from reprforge.policy_replay import load_replay_data


def _bank(path: Path, suffix: str) -> None:
    path.mkdir()
    item_id = f"item-{suffix}"
    query_id = f"query-{suffix}"
    (path / "items.jsonl").write_text(
        json.dumps(
            {
                "item_id": item_id,
                "content_type": "text",
                "route_costs": {
                    "image": {"index_bytes": 8, "encode_ms": 2.0},
                    "text": {"index_bytes": 4, "encode_ms": 1.0},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "queries.jsonl").write_text(
        json.dumps(
            {
                "query_id": query_id,
                "relevance": {item_id: 1.0},
                "candidate_item_ids": [item_id],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "scores.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "route": route,
                    "query_id": query_id,
                    "item_id": item_id,
                    "score": score,
                }
            )
            for route, score in (("text", 1.0), ("image", 2.0))
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "dataset_revision": "dataset-sha",
                "model": "fixture-model",
                "model_revision": "model-sha",
                "documents": [
                    {
                        "document_index": int(suffix),
                        "domain": f"domain-{suffix}",
                    }
                ],
                "embeddings_persisted": False,
            }
        ),
        encoding="utf-8",
    )


def test_merge_reuses_compatible_route_banks(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _bank(first, "1")
    _bank(second, "2")
    output = tmp_path / "merged"
    manifest = merge_replay_banks([first, second], output)
    data = load_replay_data(
        output / "items.jsonl",
        output / "queries.jsonl",
        output / "scores.jsonl",
    )
    assert manifest["document_indices"] == [1, 2]
    assert manifest["layout_count"] == 2
    assert len(data.items) == 2
    assert set(json.loads((output / "baselines.json").read_text())) == {
        "fixed-hybrid-image",
        "uniform-image",
        "uniform-text",
    }
